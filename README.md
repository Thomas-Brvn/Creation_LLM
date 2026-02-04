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

## Partie 3 : Attention (concept)

Le mot "**il**" dans "Le chat dort car **il** est fatigué" fait référence à "chat". Comment le modèle le sait-il ? Grâce à l'**attention** : chaque token regarde les autres tokens pour comprendre le contexte.

```
"il" regarde → ["Le", "chat", "dort", "car"] → comprend que "il" = "chat"
```

L'attention calcule un **score de pertinence** entre chaque paire de tokens, puis fait une moyenne pondérée.

<details>
<summary><strong>📖 Voir les détails complets sur l'Attention</strong></summary>

---

### Pourquoi l'attention ?

Sans contexte, les mots sont ambigus :

```
"La souris mange le fromage"  → souris = animal
"La souris ne marche plus"    → souris = périphérique
```

L'attention permet au modèle de **regarder les autres mots** pour lever l'ambiguïté.

---

### Self-Attention

"Self" car les tokens d'une même séquence s'observent entre eux :

```
Séquence: ["Le", "chat", "mange"]

"Le"    regarde: ["Le", "chat", "mange"]
"chat"  regarde: ["Le", "chat", "mange"]
"mange" regarde: ["Le", "chat", "mange"]
```

Chaque token calcule à quel point les autres tokens sont **pertinents** pour lui.

---

### Intuition : la fête

Imagine une fête avec 4 personnes. Tu veux savoir à qui parler :

1. Tu regardes chaque personne (calcul des scores)
2. Tu décides qui est intéressant pour toi (scores de pertinence)
3. Tu écoutes plus ceux qui t'intéressent (moyenne pondérée)

```
Toi → [Alice: 0.5, Bob: 0.3, Claire: 0.2]
        ↓
Tu absorbes 50% d'Alice, 30% de Bob, 20% de Claire
```

C'est exactement ce que fait l'attention avec les tokens.

---

### Masquage causal (GPT)

Dans un LLM comme GPT, un token ne peut voir que les tokens **précédents** (pas le futur) :

```
Séquence: ["Le", "chat", "mange", "la", "souris"]

"Le"     voit: ["Le"]
"chat"   voit: ["Le", "chat"]
"mange"  voit: ["Le", "chat", "mange"]
"la"     voit: ["Le", "chat", "mange", "la"]
"souris" voit: ["Le", "chat", "mange", "la", "souris"]
```

Pourquoi ? Sinon le modèle "tricherait" en regardant la réponse pendant l'entraînement.

---

### Complexité O(n²)

Chaque token regarde **tous** les autres tokens :

```
n tokens → n × n = n² comparaisons

 64 tokens  →    4,096 comparaisons
256 tokens  →   65,536 comparaisons
1024 tokens → 1,048,576 comparaisons
```

C'est pourquoi les LLMs ont une limite de contexte (`max_seq_len`).

---

### En résumé

```
Embeddings (n, d_model)
         ↓
    Self-Attention  →  Chaque token regarde les autres
         ↓
Contexte enrichi (n, d_model)
```

</details>

### Questions de vérification

1. Pourquoi "il" a besoin de regarder les autres mots ?
2. Dans GPT, le 3ème token peut-il voir le 5ème token ?
3. Pourquoi la complexité est O(n²) ?

---

## Partie 4 : Attention (calculs)

L'attention utilise trois vecteurs par token : **Query** (ce que je cherche), **Key** (ce que je contiens), **Value** (l'info que je donne). La formule :

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d_k) × V
```

En gros : on calcule la similarité entre Q et K, on normalise avec softmax, puis on fait une moyenne pondérée des V.

<details>
<summary><strong>📖 Voir les détails complets sur Q, K, V</strong></summary>

---

### Query, Key, Value - Intuition

Imagine une bibliothèque :

| Concept | Analogie | Rôle |
|---------|----------|------|
| **Query (Q)** | Ta question | "Je cherche des infos sur les chats" |
| **Key (K)** | Titre du livre | "Animaux domestiques", "Cuisine", ... |
| **Value (V)** | Contenu du livre | L'information utile |

Tu compares ta **question** (Q) avec les **titres** (K) pour trouver les livres pertinents, puis tu lis leur **contenu** (V).

---

### Comment obtenir Q, K, V ?

Chaque token a un embedding. On le projette avec 3 matrices apprises :

```
embedding (d_model) → W_Q → Query  (d_k)
embedding (d_model) → W_K → Key    (d_k)
embedding (d_model) → W_V → Value  (d_v)
```

```python
Q = embedding @ W_Q  # (n, d_model) @ (d_model, d_k) → (n, d_k)
K = embedding @ W_K  # (n, d_model) @ (d_model, d_k) → (n, d_k)
V = embedding @ W_V  # (n, d_model) @ (d_model, d_v) → (n, d_v)
```

---

### Étape 1 : Scores d'attention

On calcule la similarité entre chaque Q et chaque K :

```
scores = Q × Kᵀ
```

```
Q: (n, d_k)
K: (n, d_k) → Kᵀ: (d_k, n)

scores = Q @ Kᵀ = (n, d_k) @ (d_k, n) = (n, n)
```

Résultat : une matrice (n × n) où `scores[i][j]` = similarité entre token i et token j.

---

### Étape 2 : Mise à l'échelle

On divise par √d_k pour stabiliser les gradients :

```
scores = scores / √d_k
```

Sans ça, les scores deviennent trop grands → softmax sature → gradients nuls.

---

### Étape 3 : Masquage causal (optionnel)

Pour GPT, on masque le futur avec -∞ :

```
scores (avant masque):       scores (après masque):
[[0.5, 0.3, 0.2]             [[0.5,  -∞,  -∞]
 [0.4, 0.6, 0.1]       →      [0.4, 0.6,  -∞]
 [0.2, 0.3, 0.5]]             [0.2, 0.3, 0.5]]
```

---

### Étape 4 : Softmax

On convertit les scores en probabilités (somme = 1 par ligne) :

```
weights = softmax(scores)
```

```
scores: [2.0, 1.0, -∞]  →  weights: [0.73, 0.27, 0.00]
```

---

### Étape 5 : Moyenne pondérée des Values

```
output = weights × V
```

```
weights: (n, n)
V: (n, d_v)

output = weights @ V = (n, n) @ (n, d_v) = (n, d_v)
```

Chaque token obtient un mélange des Values des autres tokens.

---

### Formule complète

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d_k) × V
```

```
Entrée:  embeddings (n, d_model)
         ↓
      Q, K, V via projections
         ↓
      scores = Q @ Kᵀ / √d_k     → (n, n)
         ↓
      weights = softmax(scores)  → (n, n)
         ↓
      output = weights @ V       → (n, d_v)
```

---

### Exemple numérique simplifié

3 tokens, d_k = 2 :

```
Q = [[1, 0],    K = [[1, 0],    V = [[1, 2],
     [0, 1],         [0, 1],         [3, 4],
     [1, 1]]         [1, 1]]         [5, 6]]

scores = Q @ Kᵀ = [[1, 0, 1],
                   [0, 1, 1],
                   [1, 1, 2]]

scores / √2 = [[0.71, 0.00, 0.71],
               [0.00, 0.71, 0.71],
               [0.71, 0.71, 1.41]]

weights = softmax(...) ≈ [[0.39, 0.22, 0.39],
                          [0.22, 0.39, 0.39],
                          [0.26, 0.26, 0.48]]

output = weights @ V  (mélange pondéré)
```

</details>

### Questions de vérification

1. À quoi sert la division par √d_k ?
2. Quelle est la shape de la matrice de scores pour 10 tokens ?
3. Pourquoi met-on -∞ (et pas 0) pour masquer le futur ?

---

## Prochaines parties
- **Partie 5** : Multi-Head Attention
- **Partie 6** : Positional Encoding (RoPE)
- **Partie 7** : Feed-Forward, RMSNorm, résiduel
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

# Pair programming contribution
