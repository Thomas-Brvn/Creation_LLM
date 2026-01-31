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
<summary><strong>📖 Voir les détails complets sur BPE</strong></summary>

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

## Prochaines parties

- **Partie 2** : Embeddings - Convertir les IDs en vecteurs
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
