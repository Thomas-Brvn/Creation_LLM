"""
Script d'entraînement pour le mini-LLM.

Inclut:
- Boucle d'entraînement complète
- Learning rate scheduler (cosine with warmup)
- Gradient clipping
- Checkpointing
- Logging (optionnel: Weights & Biases)
"""

import os
import sys
import math
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# Ajoute le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT
from src.tokenizer import BPETokenizer
from src.dataset import TextDataset, create_dataloaders, download_sample_data, create_dummy_data
from configs.default import ModelConfig, TrainingConfig


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1
) -> LambdaLR:
    """
    Crée un scheduler cosine avec warmup.

    Le learning rate:
    1. Monte linéairement de 0 à lr_max pendant warmup_steps
    2. Descend en cosine jusqu'à min_lr_ratio * lr_max
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Warmup linéaire
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    tokenizer: BPETokenizer,
    step: int,
    loss: float,
    config: ModelConfig,
    path: str
):
    """Sauvegarde un checkpoint complet."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "loss": loss,
        "config": config.__dict__,
    }
    torch.save(checkpoint, path)

    # Sauvegarde le tokenizer à côté
    tokenizer_path = path.replace(".pt", "_tokenizer.json")
    tokenizer.save(tokenizer_path)

    print(f"Checkpoint sauvegardé: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None
) -> dict:
    """Charge un checkpoint."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Checkpoint chargé: {path} (step {checkpoint['step']})")
    return checkpoint


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 50
) -> float:
    """Évalue le modèle sur un dataset."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if n_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        output = model(input_ids, labels=labels)
        total_loss += output["loss"].item()
        n_batches += 1

    model.train()
    return total_loss / n_batches if n_batches > 0 else float("inf")


def train(
    model_config: ModelConfig,
    train_config: TrainingConfig,
    data_path: str,
    val_path: str | None = None,
    resume_from: str | None = None,
    use_wandb: bool = False
):
    """
    Boucle d'entraînement principale.

    Args:
        model_config: Configuration du modèle
        train_config: Configuration de l'entraînement
        data_path: Chemin vers les données d'entraînement
        val_path: Chemin vers les données de validation
        resume_from: Chemin vers un checkpoint pour reprendre
        use_wandb: Utiliser Weights & Biases pour le logging
    """
    # Device
    if train_config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Utilisation GPU: {torch.cuda.get_device_name(0)}")
    elif train_config.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Utilisation Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("Utilisation CPU")

    # Seed pour reproductibilité
    torch.manual_seed(train_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(train_config.seed)

    # Crée le dossier de checkpoints
    os.makedirs(train_config.checkpoint_path, exist_ok=True)

    # ===== TOKENIZER =====
    print("\n=== Préparation du tokenizer ===")
    tokenizer_path = os.path.join(train_config.checkpoint_path, "tokenizer.json")

    if os.path.exists(tokenizer_path):
        print(f"Chargement du tokenizer existant: {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print("Entraînement d'un nouveau tokenizer...")
        tokenizer = BPETokenizer(vocab_size=model_config.vocab_size)

        # Charge le texte d'entraînement
        with open(data_path, "r", encoding="utf-8") as f:
            train_text = f.read()

        tokenizer.train(train_text, verbose=True)
        tokenizer.save(tokenizer_path)

    # Met à jour la taille du vocabulaire dans la config
    model_config.vocab_size = len(tokenizer)
    print(f"Taille du vocabulaire: {model_config.vocab_size}")

    # ===== DATASET =====
    print("\n=== Préparation des données ===")
    train_loader, val_loader = create_dataloaders(
        train_path=data_path,
        val_path=val_path,
        tokenizer=tokenizer,
        batch_size=train_config.batch_size,
        max_seq_len=model_config.max_seq_len,
        num_workers=0  # 0 pour éviter les problèmes sur Mac
    )

    total_steps = len(train_loader) * train_config.n_epochs
    print(f"Nombre total de steps: {total_steps}")

    # ===== MODÈLE =====
    print("\n=== Création du modèle ===")
    model = GPT(model_config).to(device)

    # ===== OPTIMISEUR =====
    # Sépare les paramètres avec/sans weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer = AdamW([
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=train_config.learning_rate, betas=(0.9, 0.95))

    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=train_config.warmup_steps,
        total_steps=total_steps
    )

    # ===== REPRISE =====
    start_step = 0
    if resume_from:
        checkpoint = load_checkpoint(resume_from, model, optimizer, scheduler)
        start_step = checkpoint["step"]

    # ===== WANDB =====
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="mini-llm",
                config={
                    "model": model_config.__dict__,
                    "training": train_config.__dict__
                }
            )
            wandb.watch(model, log="gradients", log_freq=100)
        except ImportError:
            print("wandb non installé, logging désactivé")
            use_wandb = False

    # ===== ENTRAÎNEMENT =====
    print("\n=== Début de l'entraînement ===")
    model.train()

    global_step = start_step
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(train_config.n_epochs):
        epoch_loss = 0.0
        epoch_tokens = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_config.n_epochs}")

        for batch_idx, batch in enumerate(pbar):
            global_step += 1

            # Déplace sur device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            output = model(input_ids, labels=labels)
            loss = output["loss"]

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.max_grad_norm
            )

            # Update
            optimizer.step()
            scheduler.step()

            # Stats
            epoch_loss += loss.item()
            epoch_tokens += input_ids.numel()

            # Logging
            if global_step % train_config.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                tokens_per_sec = epoch_tokens / (time.time() - start_time)

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr:.2e}",
                    "tok/s": f"{tokens_per_sec:.0f}"
                })

                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item(),
                        "train/tokens_per_sec": tokens_per_sec,
                        "step": global_step
                    })

            # Évaluation
            if val_loader and global_step % train_config.eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"\nValidation loss: {val_loss:.4f}")

                if use_wandb:
                    wandb.log({"val/loss": val_loss, "step": global_step})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, scheduler, tokenizer,
                        global_step, val_loss, model_config,
                        os.path.join(train_config.checkpoint_path, "best.pt")
                    )

            # Sauvegarde périodique
            if global_step % train_config.save_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, tokenizer,
                    global_step, loss.item(), model_config,
                    os.path.join(train_config.checkpoint_path, f"step_{global_step}.pt")
                )

        # Fin d'epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} terminée - Loss moyenne: {avg_loss:.4f}")

    # Sauvegarde finale
    save_checkpoint(
        model, optimizer, scheduler, tokenizer,
        global_step, loss.item(), model_config,
        os.path.join(train_config.checkpoint_path, "final.pt")
    )

    print("\n=== Entraînement terminé ===")
    print(f"Temps total: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Checkpoints sauvegardés dans: {train_config.checkpoint_path}")

    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Entraîne un mini-LLM from scratch")

    parser.add_argument("--data", type=str, default=None,
                        help="Chemin vers les données d'entraînement")
    parser.add_argument("--val-data", type=str, default=None,
                        help="Chemin vers les données de validation")
    parser.add_argument("--resume", type=str, default=None,
                        help="Chemin vers un checkpoint pour reprendre")
    parser.add_argument("--wandb", action="store_true",
                        help="Utiliser Weights & Biases")

    # Config modèle
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--max-seq-len", type=int, default=256)

    # Config entraînement
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="mps",
                        choices=["cuda", "mps", "cpu"])

    args = parser.parse_args()

    # Crée les configs
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len
    )

    train_config = TrainingConfig(
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device
    )

    # Données
    if args.data is None:
        print("Pas de données spécifiées, téléchargement de données d'exemple...")
        data_path = download_sample_data("data")
    else:
        data_path = args.data

    # Lance l'entraînement
    train(
        model_config=model_config,
        train_config=train_config,
        data_path=data_path,
        val_path=args.val_data,
        resume_from=args.resume,
        use_wandb=args.wandb
    )


if __name__ == "__main__":
    main()
