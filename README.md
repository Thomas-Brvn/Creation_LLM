# Créer un LLM from Scratch

Projet éducatif pour comprendre comment fonctionne un Large Language Model (LLM) de type GPT, étape par étape.

---

## Introduction

### C'est quoi un LLM ?

Un LLM (Large Language Model) est un modèle d'intelligence artificielle capable de comprendre et générer du texte. ChatGPT, Claude, LLaMA sont des exemples de LLMs.

### Comment ça fonctionne ?

Le principe est simple : **prédire le mot suivant**.

```
Entrée:  "Le chat mange la"
Sortie:  "souris" (prédiction)
```

En répétant cette prédiction, le modèle génère du texte :

```
"Le chat" → "mange"
"Le chat mange" → "la"
"Le chat mange la" → "souris"
...
```

### Le pipeline complet

```
Texte brut
    ↓
1. TOKENIZATION  →  Convertir texte en nombres
    ↓
2. EMBEDDING     →  Convertir nombres en vecteurs
    ↓
3. ATTENTION     →  Comprendre le contexte
    ↓
4. GÉNÉRATION    →  Prédire le prochain token
```

---

## Partie 1 : Tokenization (BPE)

Un modèle ne comprend pas le texte, seulement des **nombres**. La tokenization convertit le texte en IDs :

```
"Bonjour le monde" → [456, 12, 892] → Le modèle
```

**BPE (Byte Pair Encoding)** fusionne les caractères fréquents pour créer un vocabulaire efficace. Avantage : aucun mot n'est "inconnu", tout peut être tokenizé.

<details>
<summary><strong>Détails complets sur BPE</strong></summary>

---

### Les 256 bytes de base

```
1 byte = 8 bits = 2⁸ = 256 valeurs possibles (0 à 255)
```

C'est la base de l'informatique. Chaque caractère a un code :

```
"A" = 65    "a" = 97    " " = 32
"é" = 195 + 169 (2 bytes en UTF-8)
```

Tout texte peut être représenté en bytes → BPE peut tokenizer **n'importe quel texte**.

---

### L'algorithme BPE

**Idée :** Fusionner les paires de caractères les plus fréquentes.

**Exemple** avec `"abab abab"` :

```
Départ:  [a, b, a, b, ' ', a, b, a, b]  →  9 tokens

Paire (a,b) apparaît 4 fois → fusion en "ab"
         [ab, ab, ' ', ab, ab]          →  5 tokens

Paire (ab,ab) apparaît 2 fois → fusion en "abab"
         [abab, ' ', abab]              →  3 tokens
```

**Résultat :** 9 tokens → 3 tokens !

---

### vocab_size

```
vocab_size = 256 (bytes) + 4 (spéciaux) + nombre de merges
```

| vocab_size | Séquences | Modèle |
|------------|-----------|--------|
| Petit (1K) | Longues | Léger |
| Grand (50K) | Courtes | Lourd |

---

### Tokens spéciaux

| Token | Rôle |
|-------|------|
| `<pad>` | Remplissage |
| `<unk>` | Mot inconnu (jamais utilisé avec BPE) |
| `<bos>` | Début de séquence |
| `<eos>` | Fin de séquence |

---

### BPE vs ancien système

```
Ancien:  "quinoa" → <UNK>  (mot inconnu !)
BPE:     "quinoa" → [qui][no][a]  (toujours découpable)
```

</details>

### Questions de vérification

1. Pourquoi exactement 256 bytes de base ?
2. Un mot inventé "xkzbrt" génère-t-il une erreur avec BPE ?
3. Si j'augmente vocab_size, les séquences sont plus courtes ou plus longues ?

---

## Partie 2 : Embeddings

Les IDs de tokens sont juste des indices (456, 12, 892...). Le modèle a besoin de **vecteurs riches** pour capturer le sens. L'embedding convertit chaque ID en un vecteur de dimension `d_model` :

```
Token ID 456 ("chat") → [0.2, -0.5, 0.8, ..., 0.1]  (384 dimensions)
```

Ces vecteurs sont **appris** pendant l'entraînement : les mots similaires finissent proches dans l'espace vectoriel.

<details>
<summary><strong>📖 Voir les détails complets sur les Embeddings</strong></summary>

---

### La table d'embedding

C'est une matrice de taille `vocab_size × d_model` :

```
vocab_size = 8192 tokens
d_model = 384 dimensions

Table: 8192 × 384 = 3,145,728 paramètres
```

Chaque ligne correspond à un token :

```
ID 0   → [0.1, 0.3, -0.2, ...]   (ligne 0)
ID 1   → [0.5, -0.1, 0.7, ...]   (ligne 1)
...
ID 456 → [0.2, -0.5, 0.8, ...]   (ligne 456 = "chat")
```

---

### Lookup (recherche)

L'embedding est juste une recherche dans la table :

```python
# Pseudo-code
embedding_table = matrix[vocab_size, d_model]

def embed(token_id):
    return embedding_table[token_id]  # Retourne la ligne
```

```
Entrée:  [456, 12, 892]  (3 token IDs)
Sortie:  [[...], [...], [...]]  (3 vecteurs de 384 dim)
         → Tensor de shape (3, 384)
```

---

### Pourquoi d_model ?

`d_model` = dimension des vecteurs dans tout le modèle.

| d_model | Capacité | Paramètres |
|---------|----------|------------|
| 128 | Faible | Léger |
| 384 | Moyenne | ~10M params |
| 768 | Haute | ~100M params |
| 4096 | Très haute | GPT-3 scale |

Plus `d_model` est grand, plus le modèle peut encoder d'information par token.

---

### Propriété : mots similaires = vecteurs proches

Après entraînement, les embeddings capturent le sens :

```
distance("roi", "reine") < distance("roi", "voiture")
distance("chat", "chien") < distance("chat", "avion")
```

On peut même faire de l'arithmétique :

```
embedding("roi") - embedding("homme") + embedding("femme") ≈ embedding("reine")
```

---

### En résumé

```
Token IDs        →  Embedding Table  →  Vecteurs
[456, 12, 892]   →  lookup           →  (3, 384)
```

</details>

### Questions de vérification

1. Quelle est la taille de la table d'embedding si vocab_size=4096 et d_model=256 ?
2. L'embedding est-il appris ou fixé à l'avance ?
3. Pourquoi les mots similaires ont-ils des vecteurs proches ?

---

## Prochaines parties
- **Partie 3** : Attention (concept) - Comprendre pourquoi chaque mot regarde les autres
- **Partie 4** : Attention (calculs) - Les maths derrière Q, K, V
- **Partie 5** : Multi-Head Attention - Plusieurs "points de vue"
- **Partie 6** : Positional Encoding - Comment le modèle connaît l'ordre des mots
- **Partie 7** : Feed-Forward et Normalisation
- **Partie 8** : Architecture GPT complète
- **Partie 9** : Entraînement
- **Partie 10** : Génération de texte

---

## Structure du projet

```
Creation_LLM/
├── README.md              ← Ce fichier
├── src/
│   ├── tokenizer.py       ← Implémentation BPE
│   ├── model.py           ← Architecture Transformer
│   ├── train.py           ← Script d'entraînement
│   └── generate.py        ← Script de génération
├── docs/
│   └── CONCEPTS_LLM.md    ← Récapitulatif technique
└── data/                  ← Données d'entraînement
```

---

## Ressources

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention Is All You Need (paper)](https://arxiv.org/abs/1706.03762)
