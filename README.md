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

### Pourquoi tokenizer ?

Un modèle de deep learning ne comprend pas le texte. Il ne comprend que des **nombres**.

```
"Bonjour" → ??? → Le modèle
```

La tokenization convertit le texte en une liste de nombres (IDs).

```
"Bonjour" → [2, 456, 3] → Le modèle
```

---

### Les 256 bytes de base

#### Pourquoi 256 ?

```
1 byte = 8 bits = 2⁸ = 256 valeurs possibles (0 à 255)
```

C'est la base de l'informatique. **Non modifiable.**

Chaque caractère a un code :

```
"A" = 65
"a" = 97
" " = 32
"é" = 195 + 169 (2 bytes en UTF-8)
```

#### Conséquence importante

Tout texte peut être représenté en bytes → BPE peut tokenizer **n'importe quel texte**, même des mots inventés.

---

### L'algorithme BPE (Byte Pair Encoding)

#### Idée

Fusionner les paires de caractères les plus fréquentes pour créer de nouveaux tokens.

#### Exemple pas à pas

Texte : `"abab abab"`

**Étape 0 : Chaque caractère = 1 token**

```
[a, b, a, b, ' ', a, b, a, b]  →  9 tokens
```

**Étape 1 : Compter les paires**

```
(a, b) → 4 fois  ← LA PLUS FRÉQUENTE
(b, a) → 2 fois
(b, ' ') → 1 fois
(' ', a) → 1 fois
```

**Étape 2 : Fusionner la paire (a, b) → nouveau token "ab"**

```
[ab, ab, ' ', ab, ab]  →  5 tokens
```

**Étape 3 : Recompter les paires**

```
(ab, ab) → 2 fois  ← LA PLUS FRÉQUENTE
(ab, ' ') → 1 fois
(' ', ab) → 1 fois
```

**Étape 4 : Fusionner (ab, ab) → nouveau token "abab"**

```
[abab, ' ', abab]  →  3 tokens
```

**Résultat :** 9 tokens → 3 tokens !

---

### vocab_size

C'est le nombre total de tokens dans le vocabulaire :

```
vocab_size = 256 (bytes) + 4 (spéciaux) + nombre de merges
```

#### Impact du vocab_size

| vocab_size | Séquences | Modèle | Contexte effectif |
|------------|-----------|--------|-------------------|
| Petit (1000) | Longues | Léger | Moins de mots visibles |
| Grand (50000) | Courtes | Lourd | Plus de mots visibles |

#### Exemple

```
Texte: "anticonstitutionnellement"

vocab_size = 1000:  [an][ti][con][sti][tu][tion][nel][le][ment]  = 9 tokens
vocab_size = 50000: [anticonstitutionnellement]                  = 1 token
```

---

### Tokens spéciaux

| Token | ID | Rôle |
|-------|-----|------|
| `<pad>` | 0 | Remplissage pour égaliser les longueurs |
| `<unk>` | 1 | Mot inconnu (jamais utilisé avec BPE) |
| `<bos>` | 2 | Début de séquence (Beginning Of Sequence) |
| `<eos>` | 3 | Fin de séquence (End Of Sequence) |

---

### BPE vs ancien système

#### Ancien système (par mots)

```
Vocabulaire fixe : ["le", "chat", "mange", ...]

"quinoa" → <UNK>  (mot inconnu !)
```

#### BPE (moderne)

```
"quinoa" → [qui][no][a]  (découpé, jamais inconnu !)
"xyzabc" → [x][y][z][a][b][c]  (fonctionne toujours)
```

**Avantage majeur de BPE :** Pas de mot inconnu, tout peut être tokenizé.

---

### Questions de vérification - Partie 1

1. Pourquoi exactement 256 bytes de base ?
2. Un mot inventé "xkzbrt" génère-t-il une erreur avec BPE ?
3. Si j'augmente vocab_size, les séquences sont plus courtes ou plus longues ?
4. Pourquoi BPE n'a jamais de token `<UNK>` ?

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
