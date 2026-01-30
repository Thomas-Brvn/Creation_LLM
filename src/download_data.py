"""
Script pour télécharger et préparer Wikipedia français.

Usage:
    python src/download_data.py --size small   # ~100MB, rapide
    python src/download_data.py --size medium  # ~500MB
    python src/download_data.py --size large   # ~2GB
"""

import os
import argparse
from tqdm import tqdm


def download_wikipedia_fr(output_dir: str = "data", size: str = "small"):
    """
    Télécharge un sous-ensemble de Wikipedia français.

    Args:
        output_dir: Dossier de sortie
        size: "small" (~100MB), "medium" (~500MB), "large" (~2GB)
    """
    from datasets import load_dataset

    # Nombre d'articles selon la taille
    n_articles = {
        "small": 50_000,      # ~100MB, ~5min
        "medium": 250_000,    # ~500MB, ~20min
        "large": 1_000_000,   # ~2GB, ~1h
    }

    n = n_articles.get(size, 50_000)

    print(f"Téléchargement de Wikipedia FR ({size}: {n:,} articles)...")
    print("Cela peut prendre quelques minutes...\n")

    # Charge le dataset Wikipedia français
    # On utilise le streaming pour ne pas tout charger en mémoire
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.fr",  # Version novembre 2023
        split="train",
        streaming=True
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"wikipedia_fr_{size}.txt")

    # Écrit les articles dans un fichier texte
    total_chars = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, article in enumerate(tqdm(dataset, total=n, desc="Téléchargement")):
            if i >= n:
                break

            # Extrait le texte de l'article
            title = article["title"]
            text = article["text"]

            # Ignore les articles trop courts
            if len(text) < 500:
                continue

            # Écrit avec un séparateur
            f.write(f"# {title}\n\n")
            f.write(text)
            f.write("\n\n---\n\n")

            total_chars += len(text)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ Téléchargement terminé!")
    print(f"  Fichier: {output_path}")
    print(f"  Taille: {size_mb:.1f} MB")
    print(f"  Caractères: {total_chars:,}")

    return output_path


def download_french_books(output_dir: str = "data", n_books: int = 100):
    """
    Télécharge des livres français du domaine public (Project Gutenberg).
    Alternative plus légère à Wikipedia.
    """
    from datasets import load_dataset

    print(f"Téléchargement de livres français ({n_books} livres)...")

    # Dataset de livres en français
    dataset = load_dataset(
        "pg19",  # Project Gutenberg
        split="train",
        streaming=True
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "french_books.txt")

    # Filtre les livres en français (heuristique basique)
    french_words = {"le", "la", "les", "de", "du", "des", "un", "une", "et", "est", "que", "qui"}

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for book in tqdm(dataset, desc="Recherche de livres français"):
            text = book["text"][:1000].lower()  # Premiers 1000 chars

            # Compte les mots français
            words = set(text.split())
            french_count = len(words & french_words)

            if french_count >= 5:  # Probablement français
                f.write(book["text"])
                f.write("\n\n---\n\n")
                count += 1

                if count >= n_books:
                    break

    print(f"\n✓ {count} livres français téléchargés: {output_path}")
    return output_path


def create_sample_dataset(output_dir: str = "data"):
    """
    Crée un petit dataset d'exemple pour tester rapidement.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample_fr.txt")

    # Textes d'exemple variés
    texts = [
        # Contes
        """Il était une fois, dans un petit village au pied des montagnes, une jeune fille nommée Marie.
        Elle vivait avec sa grand-mère dans une petite maison au toit de chaume. Chaque matin, elle
        allait chercher de l'eau à la fontaine du village et cueillait des fleurs sauvages dans les prés.""",

        # Science
        """L'intelligence artificielle est un domaine de l'informatique qui vise à créer des machines
        capables de simuler l'intelligence humaine. Les réseaux de neurones artificiels sont inspirés
        du fonctionnement du cerveau humain. Ils sont composés de neurones interconnectés qui traitent
        l'information de manière parallèle.""",

        # Histoire
        """La Révolution française de 1789 a profondément transformé la société française. La prise
        de la Bastille le 14 juillet marque le début d'une nouvelle ère. Les idées des Lumières,
        portées par des philosophes comme Voltaire et Rousseau, ont inspiré ce mouvement.""",

        # Technologie
        """Les ordinateurs modernes utilisent des processeurs composés de milliards de transistors.
        Ces minuscules composants électroniques peuvent effectuer des calculs à une vitesse
        incroyable. Le développement de l'informatique a révolutionné notre façon de communiquer
        et de travailler.""",

        # Nature
        """La forêt amazonienne est le poumon de notre planète. Elle abrite une biodiversité
        exceptionnelle avec des millions d'espèces animales et végétales. Les arbres géants
        peuvent atteindre plus de cinquante mètres de hauteur et vivre plusieurs siècles.""",

        # Cuisine
        """La cuisine française est reconnue dans le monde entier pour sa finesse et sa diversité.
        Chaque région possède ses spécialités : le cassoulet du Sud-Ouest, la choucroute alsacienne,
        les crêpes bretonnes ou encore la bouillabaisse marseillaise.""",

        # Littérature
        """Victor Hugo est l'un des plus grands écrivains de la littérature française. Son roman
        Les Misérables raconte l'histoire de Jean Valjean, un ancien bagnard en quête de rédemption.
        Cette œuvre magistrale dépeint la société française du dix-neuvième siècle.""",

        # Philosophie
        """René Descartes, philosophe et mathématicien français, est célèbre pour sa formule
        'Je pense, donc je suis'. Cette affirmation constitue le fondement de sa philosophie
        et marque le début de la pensée moderne. Le doute méthodique est au cœur de sa démarche.""",

        # Musique
        """La musique classique française a produit de nombreux compositeurs de génie. Claude Debussy
        a révolutionné l'harmonie avec ses œuvres impressionnistes. Maurice Ravel est connu pour
        son célèbre Boléro, une pièce orchestrale hypnotique.""",

        # Sport
        """Le Tour de France est la plus grande course cycliste du monde. Chaque été, les meilleurs
        coureurs s'affrontent sur les routes de France pendant trois semaines. Les étapes de montagne
        dans les Alpes et les Pyrénées sont les plus spectaculaires.""",
    ]

    # Répète et mélange pour avoir plus de données
    import random
    all_texts = texts * 500  # ~50KB
    random.shuffle(all_texts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_texts))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"✓ Dataset d'exemple créé: {output_path} ({size_kb:.1f} KB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Télécharge des données d'entraînement")
    parser.add_argument("--source", type=str, default="wikipedia",
                        choices=["wikipedia", "books", "sample"],
                        help="Source des données")
    parser.add_argument("--size", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Taille du dataset (pour wikipedia)")
    parser.add_argument("--output", type=str, default="data",
                        help="Dossier de sortie")

    args = parser.parse_args()

    if args.source == "wikipedia":
        download_wikipedia_fr(args.output, args.size)
    elif args.source == "books":
        download_french_books(args.output)
    else:
        create_sample_dataset(args.output)


if __name__ == "__main__":
    main()
