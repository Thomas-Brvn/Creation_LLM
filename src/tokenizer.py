"""
Tokenizer BPE (Byte Pair Encoding) implémenté from scratch.

Le BPE est l'algorithme de tokenization utilisé par GPT-2/3/4.
Il fonctionne en:
1. Commençant avec des bytes individuels (256 tokens de base)
2. Fusionnant itérativement les paires de tokens les plus fréquentes
3. Construisant ainsi un vocabulaire de sous-mots
"""

import json
import os
import regex as re
from collections import defaultdict
from typing import Optional

# Pattern de pré-tokenization inspiré de GPT-2
# Sépare le texte en mots, nombres, ponctuation, etc.
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    """
    Tokenizer Byte Pair Encoding from scratch.

    Attributs:
        vocab_size: Taille cible du vocabulaire
        merges: Liste des paires de tokens fusionnées
        vocab: Dictionnaire token -> id
        inverse_vocab: Dictionnaire id -> token
    """

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[bytes, int] = {}
        self.inverse_vocab: dict[int, bytes] = {}

        # Tokens spéciaux
        self.special_tokens = {
            "<|pad|>": 0,
            "<|unk|>": 1,
            "<|bos|>": 2,
            "<|eos|>": 3,
        }

        self.pattern = re.compile(GPT2_PATTERN)
        self._init_base_vocab()

    def _init_base_vocab(self):
        """Initialise le vocabulaire de base avec les 256 bytes + tokens spéciaux."""
        # D'abord les tokens spéciaux
        for token, idx in self.special_tokens.items():
            self.vocab[token.encode("utf-8")] = idx
            self.inverse_vocab[idx] = token.encode("utf-8")

        # Puis les 256 bytes possibles
        for i in range(256):
            byte = bytes([i])
            idx = i + len(self.special_tokens)
            self.vocab[byte] = idx
            self.inverse_vocab[idx] = byte

    def _get_stats(self, token_ids: list[list[int]]) -> dict[tuple[int, int], int]:
        """Compte la fréquence de chaque paire de tokens adjacents."""
        stats = defaultdict(int)
        for ids in token_ids:
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                stats[pair] += 1
        return stats

    def _merge(
        self, token_ids: list[list[int]], pair: tuple[int, int], new_id: int
    ) -> list[list[int]]:
        """Fusionne toutes les occurrences d'une paire en un nouveau token."""
        new_token_ids = []
        for ids in token_ids:
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            new_token_ids.append(new_ids)
        return new_token_ids

    def train(self, text: str, verbose: bool = True):
        """
        Entraîne le tokenizer BPE sur un corpus de texte.

        Args:
            text: Texte d'entraînement
            verbose: Afficher la progression
        """
        # Pré-tokenization: sépare le texte en chunks
        chunks = self.pattern.findall(text)

        # Convertit chaque chunk en liste de bytes (IDs de base)
        token_ids = []
        for chunk in chunks:
            chunk_bytes = chunk.encode("utf-8")
            ids = [self.vocab[bytes([b])] for b in chunk_bytes]
            token_ids.append(ids)

        # Nombre de merges à effectuer
        n_merges = self.vocab_size - len(self.vocab)

        if verbose:
            print(f"Entraînement BPE: {n_merges} merges à effectuer")
            print(f"Vocabulaire initial: {len(self.vocab)} tokens")

        for i in range(n_merges):
            # Trouve la paire la plus fréquente
            stats = self._get_stats(token_ids)
            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            best_count = stats[best_pair]

            if best_count < 2:
                # Pas assez d'occurrences pour justifier un merge
                break

            # Crée un nouveau token
            new_id = len(self.vocab)
            self.merges[best_pair] = new_id

            # Concatène les bytes des deux tokens
            new_token = self.inverse_vocab[best_pair[0]] + self.inverse_vocab[best_pair[1]]
            self.vocab[new_token] = new_id
            self.inverse_vocab[new_id] = new_token

            # Applique le merge
            token_ids = self._merge(token_ids, best_pair, new_id)

            if verbose and (i + 1) % 500 == 0:
                print(f"Merge {i + 1}/{n_merges}: {best_pair} -> {new_id} (freq: {best_count})")

        if verbose:
            print(f"Vocabulaire final: {len(self.vocab)} tokens")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode un texte en liste d'IDs de tokens.

        Args:
            text: Texte à encoder
            add_special_tokens: Ajouter <|bos|> et <|eos|>

        Returns:
            Liste d'IDs de tokens
        """
        # Pré-tokenization
        chunks = self.pattern.findall(text)

        all_ids = []
        if add_special_tokens:
            all_ids.append(self.special_tokens["<|bos|>"])

        for chunk in chunks:
            # Convertit en bytes
            chunk_bytes = chunk.encode("utf-8")
            ids = [self.vocab[bytes([b])] for b in chunk_bytes]

            # Applique tous les merges appris
            for pair, new_id in self.merges.items():
                ids = self._merge([ids], pair, new_id)[0]

            all_ids.extend(ids)

        if add_special_tokens:
            all_ids.append(self.special_tokens["<|eos|>"])

        return all_ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Décode une liste d'IDs de tokens en texte.

        Args:
            ids: Liste d'IDs de tokens
            skip_special_tokens: Ignorer les tokens spéciaux

        Returns:
            Texte décodé
        """
        special_ids = set(self.special_tokens.values())
        byte_list = []

        for id_ in ids:
            if skip_special_tokens and id_ in special_ids:
                continue
            if id_ in self.inverse_vocab:
                byte_list.append(self.inverse_vocab[id_])

        # Concatène tous les bytes et décode en UTF-8
        all_bytes = b"".join(byte_list)
        return all_bytes.decode("utf-8", errors="replace")

    def save(self, path: str):
        """Sauvegarde le tokenizer."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        data = {
            "vocab_size": self.vocab_size,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "vocab": {k.hex(): v for k, v in self.vocab.items()},
            "special_tokens": self.special_tokens,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Charge un tokenizer sauvegardé."""
        with open(path) as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.merges = {
            tuple(map(int, k.split(","))): v for k, v in data["merges"].items()
        }
        tokenizer.vocab = {bytes.fromhex(k): v for k, v in data["vocab"].items()}
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.special_tokens = data["special_tokens"]

        return tokenizer

    def __len__(self) -> int:
        return len(self.vocab)


# Test rapide
if __name__ == "__main__":
    # Exemple d'utilisation
    text = """
    Bonjour, je suis un modèle de langage créé from scratch.
    Le Byte Pair Encoding est un algorithme de tokenization très efficace.
    Il permet de représenter n'importe quel texte avec un vocabulaire fini.
    """ * 100  # Répète pour avoir assez de données

    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(text, verbose=True)

    # Test encode/decode
    test_text = "Bonjour, comment ça va ?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"\nTest:")
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Match: {test_text == decoded}")
