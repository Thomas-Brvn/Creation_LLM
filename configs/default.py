"""
Configuration par défaut pour le mini-LLM.
Architecture GPT-like avec ~10M paramètres.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration de l'architecture du modèle."""

    # Taille du vocabulaire (sera définie par le tokenizer)
    vocab_size: int = 8192

    # Dimension des embeddings et du modèle
    d_model: int = 384

    # Nombre de têtes d'attention
    n_heads: int = 6

    # Nombre de couches Transformer
    n_layers: int = 6

    # Dimension du feed-forward (généralement 4x d_model)
    d_ff: int = 1536

    # Longueur maximale de séquence
    max_seq_len: int = 256

    # Dropout
    dropout: float = 0.1

    # Bias dans les couches linéaires (GPT-2 style = True)
    bias: bool = False

    def __post_init__(self):
        """Calcule le nombre approximatif de paramètres."""
        # Embeddings
        embed_params = self.vocab_size * self.d_model
        pos_embed_params = self.max_seq_len * self.d_model

        # Par couche Transformer
        attn_params = 4 * self.d_model * self.d_model  # Q, K, V, O projections
        ff_params = 2 * self.d_model * self.d_ff  # 2 couches linéaires
        layer_norm_params = 4 * self.d_model  # 2 layer norms par couche

        layer_params = attn_params + ff_params + layer_norm_params

        # Total
        total = embed_params + pos_embed_params + (self.n_layers * layer_params) + self.vocab_size * self.d_model
        self.n_params = total
        print(f"Paramètres estimés: {total / 1e6:.2f}M")


@dataclass
class TrainingConfig:
    """Configuration de l'entraînement."""

    # Batch size
    batch_size: int = 64

    # Nombre d'epochs
    n_epochs: int = 10

    # Learning rate
    learning_rate: float = 3e-4

    # Weight decay
    weight_decay: float = 0.1

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Warmup steps
    warmup_steps: int = 100

    # Logging
    log_every: int = 10

    # Sauvegarde
    save_every: int = 1000

    # Evaluation
    eval_every: int = 500

    # Device
    device: str = "cuda"  # ou "mps" pour Mac M1/M2/M3

    # Seed
    seed: int = 42

    # Chemins
    data_path: str = "data/"
    checkpoint_path: str = "checkpoints/"
