"""
Architecture Transformer GPT-like implémentée from scratch.

Un LLM de type GPT est un "decoder-only transformer" qui:
1. Utilise l'attention causale (ne voit que les tokens précédents)
2. Prédit le prochain token de manière auto-régressive
3. Utilise des embeddings positionnels pour l'ordre des tokens
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

import sys
sys.path.append("..")
from configs.default import ModelConfig


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Plus simple et souvent aussi efficace que LayerNorm.
    Utilisé dans LLaMA et d'autres modèles récents.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    Encode les positions par rotation dans l'espace complexe.
    Avantage: meilleure généralisation aux séquences longues.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Calcul des fréquences
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pré-calcul des cos et sin
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Pré-calcule les valeurs cos et sin."""
        positions = torch.arange(seq_len)
        freqs = torch.outer(positions, self.inv_freq)
        # freqs shape: (seq_len, dim/2)

        cos = freqs.cos()
        sin = freqs.sin()

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retourne cos et sin pour la longueur de séquence donnée."""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applique RoPE à Q et K.

    Args:
        q, k: (batch, n_heads, seq_len, head_dim)
        cos, sin: (seq_len, head_dim/2)
    """
    # Sépare en deux moitiés
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    # Reshape cos et sin pour le broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim/2)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Rotation
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention avec masque causal.

    L'attention permet au modèle de "regarder" différentes parties
    de la séquence d'entrée pour chaque position de sortie.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        assert self.head_dim * config.n_heads == config.d_model, \
            "d_model doit être divisible par n_heads"

        # Projections Q, K, V, O
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Masque d'attention optionnel

        Returns:
            Tensor de shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Projections linéaires
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape pour multi-head: (batch, seq, n_heads, head_dim) -> (batch, n_heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Applique RoPE
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Scaled dot-product attention
        # scores = Q @ K^T / sqrt(d_k)
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Masque causal: empêche de voir les tokens futurs
        if mask is None:
            # Crée un masque causal triangulaire
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device),
                diagonal=1
            ).bool()
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            scores = scores.masked_fill(mask, float("-inf"))

        # Softmax et dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention @ V
        output = torch.matmul(attn_weights, v)

        # Reshape: (batch, n_heads, seq, head_dim) -> (batch, seq, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Projection de sortie
        return self.o_proj(output)


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) avec activation SwiGLU.

    SwiGLU est une variante de GLU utilisée dans LLaMA et PaLM.
    Plus efficace que ReLU ou GELU standard.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # SwiGLU utilise 3 projections au lieu de 2
        # Pour garder le même nombre de params, on ajuste la dimension
        hidden_dim = int(2 * config.d_ff / 3)
        # Arrondi au multiple de 8 le plus proche (optimisation GPU)
        hidden_dim = ((hidden_dim + 7) // 8) * 8

        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(xW1) * (xW3)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """
    Un bloc Transformer complet (Pre-LN architecture).

    Structure:
    1. RMSNorm -> Multi-Head Attention -> Residual
    2. RMSNorm -> Feed-Forward -> Residual
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ff_norm = RMSNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-LN: normalise AVANT l'opération (meilleure stabilité)
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class GPT(nn.Module):
    """
    Modèle GPT complet.

    Architecture:
    1. Token Embedding
    2. N x Transformer Blocks
    3. RMSNorm finale
    4. Language Model Head (projection vers vocabulaire)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Blocs Transformer
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Normalisation finale
        self.norm = RMSNorm(config.d_model)

        # Language Model Head (poids partagés avec embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying: partage les poids entre embedding et lm_head
        self.tok_emb.weight = self.lm_head.weight

        # Initialisation des poids
        self.apply(self._init_weights)

        # Compte les paramètres
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Nombre de paramètres: {n_params / 1e6:.2f}M")

    def _init_weights(self, module: nn.Module):
        """Initialisation des poids à la GPT-2."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) - IDs des tokens d'entrée
            labels: (batch_size, seq_len) - IDs des tokens cibles (optionnel)
            mask: Masque d'attention optionnel

        Returns:
            dict avec 'logits' et optionnellement 'loss'
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.tok_emb(input_ids)

        # Passe à travers les blocs Transformer
        for layer in self.layers:
            x = layer(x, mask)

        # Normalisation finale
        x = self.norm(x)

        # Projection vers vocabulaire
        logits = self.lm_head(x)

        output = {"logits": logits}

        # Calcul de la loss si labels fournis
        if labels is not None:
            # Shift pour prédire le prochain token
            # logits: prédiction pour position i
            # labels: token réel à position i+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100  # Ignore le padding
            )
            output["loss"] = loss

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None
    ) -> torch.Tensor:
        """
        Génère du texte de manière auto-régressive.

        Args:
            input_ids: (batch_size, seq_len) - Prompt initial
            max_new_tokens: Nombre max de tokens à générer
            temperature: Contrôle la "créativité" (1.0 = normal, <1 = conservateur, >1 = créatif)
            top_k: Garde seulement les k tokens les plus probables
            top_p: Nucleus sampling - garde les tokens jusqu'à probabilité cumulative p

        Returns:
            Tensor avec les tokens générés
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Tronque si dépasse la longueur max
            idx_cond = input_ids if input_ids.shape[1] <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            output = self(idx_cond)
            logits = output["logits"][:, -1, :]  # Prend le dernier token

            # Applique la température
            logits = logits / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Supprime les tokens avec proba cumulative > p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Échantillonne
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Concatène
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop si EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids


# Test rapide
if __name__ == "__main__":
    config = ModelConfig()
    model = GPT(config)

    # Test forward
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")

    # Test génération
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
