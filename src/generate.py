"""
Script de génération de texte avec le mini-LLM.

Permet de:
- Charger un modèle entraîné
- Générer du texte à partir d'un prompt
- Contrôler la génération (température, top-k, top-p)
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT
from src.tokenizer import BPETokenizer
from configs.default import ModelConfig


def load_model(checkpoint_path: str, device: torch.device) -> tuple[GPT, BPETokenizer]:
    """
    Charge un modèle et son tokenizer depuis un checkpoint.

    Args:
        checkpoint_path: Chemin vers le fichier .pt
        device: Device sur lequel charger le modèle

    Returns:
        (model, tokenizer)
    """
    # Charge le checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruit la config
    config_dict = checkpoint["config"]
    config = ModelConfig(**config_dict)

    # Crée et charge le modèle
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Charge le tokenizer
    tokenizer_path = checkpoint_path.replace(".pt", "_tokenizer.json")
    tokenizer = BPETokenizer.load(tokenizer_path)

    print(f"Modèle chargé: {checkpoint_path}")
    print(f"Step: {checkpoint['step']}, Loss: {checkpoint['loss']:.4f}")

    return model, tokenizer


def generate_text(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.9,
    device: torch.device = torch.device("cpu")
) -> str:
    """
    Génère du texte à partir d'un prompt.

    Args:
        model: Modèle GPT
        tokenizer: Tokenizer BPE
        prompt: Texte de départ
        max_new_tokens: Nombre max de tokens à générer
        temperature: Contrôle la créativité (0.0-2.0)
        top_k: Garde les k tokens les plus probables
        top_p: Nucleus sampling

    Returns:
        Texte généré
    """
    # Encode le prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Génère
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.special_tokens.get("<|eos|>")
        )

    # Décode
    generated_text = tokenizer.decode(output_ids[0].tolist())

    return generated_text


def interactive_mode(
    model: GPT,
    tokenizer: BPETokenizer,
    device: torch.device,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    max_tokens: int = 100
):
    """Mode interactif pour tester le modèle."""
    print("\n" + "=" * 50)
    print("Mode interactif - Mini LLM")
    print("=" * 50)
    print("Commandes:")
    print("  /quit      - Quitter")
    print("  /temp X    - Changer la température (ex: /temp 0.5)")
    print("  /topk X    - Changer top_k (ex: /topk 100)")
    print("  /topp X    - Changer top_p (ex: /topp 0.95)")
    print("  /tokens X  - Changer max_tokens (ex: /tokens 200)")
    print("=" * 50 + "\n")

    while True:
        try:
            prompt = input("\n📝 Prompt: ").strip()

            if not prompt:
                continue

            if prompt.lower() == "/quit":
                print("Au revoir!")
                break

            if prompt.startswith("/temp "):
                temperature = float(prompt.split()[1])
                print(f"Température: {temperature}")
                continue

            if prompt.startswith("/topk "):
                top_k = int(prompt.split()[1])
                print(f"Top-k: {top_k}")
                continue

            if prompt.startswith("/topp "):
                top_p = float(prompt.split()[1])
                print(f"Top-p: {top_p}")
                continue

            if prompt.startswith("/tokens "):
                max_tokens = int(prompt.split()[1])
                print(f"Max tokens: {max_tokens}")
                continue

            print("\n🤖 Génération...")
            generated = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device
            )
            print(f"\n{generated}")

        except KeyboardInterrupt:
            print("\nInterrompu. Utilisez /quit pour quitter.")
        except Exception as e:
            print(f"Erreur: {e}")


def main():
    parser = argparse.ArgumentParser(description="Génère du texte avec le mini-LLM")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Chemin vers le checkpoint du modèle")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt pour la génération (mode non-interactif)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Mode interactif")

    # Paramètres de génération
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Nombre max de tokens à générer")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Température (0.0-2.0)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling")

    # Device
    parser.add_argument("--device", type=str, default="mps",
                        choices=["cuda", "mps", "cpu"])

    args = parser.parse_args()

    # Device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Charge le modèle
    model, tokenizer = load_model(args.checkpoint, device)

    if args.interactive:
        interactive_mode(
            model, tokenizer, device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
    elif args.prompt:
        generated = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )
        print("\n" + "=" * 50)
        print("Texte généré:")
        print("=" * 50)
        print(generated)
    else:
        parser.error("Spécifiez --prompt ou --interactive")


if __name__ == "__main__":
    main()
