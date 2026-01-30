"""
Dataset et DataLoader pour l'entraînement du LLM.

Gère le chargement, la tokenization et le batching des données.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Iterator
import numpy as np
from tqdm import tqdm

from tokenizer import BPETokenizer


class TextDataset(Dataset):
    """
    Dataset pour l'entraînement d'un LLM.

    Charge un fichier texte, le tokenize, et crée des séquences
    de longueur fixe pour l'entraînement.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: BPETokenizer,
        max_seq_len: int = 256,
        stride: int | None = None
    ):
        """
        Args:
            data_path: Chemin vers le fichier texte ou dossier
            tokenizer: Tokenizer BPE entraîné
            max_seq_len: Longueur des séquences
            stride: Décalage entre séquences (défaut = max_seq_len // 2)
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len // 2

        # Charge et tokenize les données
        self.tokens = self._load_and_tokenize(data_path)

        # Crée les indices de début de chaque séquence
        self.indices = list(range(0, len(self.tokens) - max_seq_len, self.stride))

        print(f"Dataset créé: {len(self.tokens)} tokens, {len(self.indices)} séquences")

    def _load_and_tokenize(self, data_path: str) -> list[int]:
        """Charge et tokenize les données."""
        if os.path.isfile(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif os.path.isdir(data_path):
            # Concatène tous les fichiers .txt du dossier
            texts = []
            for filename in sorted(os.listdir(data_path)):
                if filename.endswith(".txt"):
                    filepath = os.path.join(data_path, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        texts.append(f.read())
            text = "\n\n".join(texts)
        else:
            raise ValueError(f"Chemin invalide: {data_path}")

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return tokens

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Retourne une séquence pour l'entraînement.

        Le modèle apprend à prédire le token suivant, donc:
        - input_ids: tokens[i:i+max_seq_len]
        - labels: tokens[i:i+max_seq_len] (même chose, le shift est fait dans le modèle)
        """
        start_idx = self.indices[idx]
        end_idx = start_idx + self.max_seq_len

        tokens = self.tokens[start_idx:end_idx]

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(tokens, dtype=torch.long)
        }


class StreamingDataset(torch.utils.data.IterableDataset):
    """
    Dataset en streaming pour de très gros fichiers.
    Ne charge pas tout en mémoire.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: BPETokenizer,
        max_seq_len: int = 256,
        buffer_size: int = 10000
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer = []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                # Tokenize la ligne
                tokens = self.tokenizer.encode(line.strip(), add_special_tokens=False)
                buffer.extend(tokens)

                # Quand le buffer est assez grand, yield des séquences
                while len(buffer) >= self.max_seq_len:
                    seq = buffer[:self.max_seq_len]
                    buffer = buffer[self.max_seq_len // 2:]  # Overlap

                    yield {
                        "input_ids": torch.tensor(seq, dtype=torch.long),
                        "labels": torch.tensor(seq, dtype=torch.long)
                    }


def create_dataloaders(
    train_path: str,
    val_path: str | None,
    tokenizer: BPETokenizer,
    batch_size: int = 32,
    max_seq_len: int = 256,
    num_workers: int = 4
) -> tuple[DataLoader, DataLoader | None]:
    """
    Crée les DataLoaders d'entraînement et validation.

    Args:
        train_path: Chemin vers les données d'entraînement
        val_path: Chemin vers les données de validation (optionnel)
        tokenizer: Tokenizer BPE
        batch_size: Taille des batches
        max_seq_len: Longueur des séquences
        num_workers: Nombre de workers pour le chargement

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = TextDataset(train_path, tokenizer, max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = None
    if val_path:
        val_dataset = TextDataset(val_path, tokenizer, max_seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


def download_sample_data(output_dir: str = "data") -> str:
    """
    Télécharge un petit dataset d'exemple pour tester.
    Utilise le dataset TinyStories ou un sous-ensemble de Wikipedia.
    """
    try:
        from datasets import load_dataset

        print("Téléchargement du dataset TinyStories...")
        dataset = load_dataset("roneneldan/TinyStories", split="train[:10000]")

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "tinystories_sample.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            for example in tqdm(dataset, desc="Écriture"):
                f.write(example["text"] + "\n\n")

        print(f"Dataset sauvegardé: {output_path}")
        return output_path

    except ImportError:
        print("Package 'datasets' non installé. Création de données d'exemple...")
        return create_dummy_data(output_dir)


def create_dummy_data(output_dir: str = "data") -> str:
    """Crée des données d'exemple si le téléchargement échoue."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample.txt")

    # Quelques textes d'exemple
    texts = [
        "Il était une fois un petit chat qui vivait dans une grande maison.",
        "Le soleil brillait ce matin-là quand Marie décida de partir en voyage.",
        "Les arbres de la forêt étaient couverts de neige blanche.",
        "Un jour, un sage rencontra un jeune homme sur la route.",
        "La mer était calme et les vagues caressaient doucement le rivage.",
        "Dans le royaume lointain, vivait un roi juste et bon.",
        "Les étoiles scintillaient dans le ciel noir de la nuit.",
        "Le petit robot apprenait chaque jour quelque chose de nouveau.",
        "La musique remplissait la salle de concert de ses notes mélodieuses.",
        "Au fond de l'océan, les poissons nageaient en bancs colorés.",
    ] * 1000  # Répète pour avoir assez de données

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(texts))

    print(f"Données d'exemple créées: {output_path}")
    return output_path


# Test
if __name__ == "__main__":
    # Crée des données d'exemple
    data_path = create_dummy_data()

    # Crée un tokenizer
    tokenizer = BPETokenizer(vocab_size=1000)
    with open(data_path, "r") as f:
        text = f.read()
    tokenizer.train(text, verbose=True)

    # Crée le dataset
    dataset = TextDataset(data_path, tokenizer, max_seq_len=128)

    # Test
    sample = dataset[0]
    print(f"\nSample:")
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"labels shape: {sample['labels'].shape}")

    # Décode pour vérifier
    decoded = tokenizer.decode(sample["input_ids"].tolist())
    print(f"Texte décodé: {decoded[:200]}...")
