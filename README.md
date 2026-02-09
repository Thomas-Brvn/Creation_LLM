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

## Partie 5 : Multi-Head Attention

Une seule attention capture un seul "point de vue". Avec **plusieurs têtes** en parallèle, le modèle peut capturer différents types de relations :

```
Tête 1 : relations syntaxiques (sujet → verbe)
Tête 2 : relations sémantiques (chat → animal)
Tête 3 : proximité (mots proches)
...
```

On divise `d_model` entre les têtes : avec 384 dimensions et 6 têtes, chaque tête travaille sur 64 dimensions.

<details>
<summary><strong>📖 Voir les détails complets sur Multi-Head Attention</strong></summary>

---

### Pourquoi plusieurs têtes ?

Une seule attention = un seul type de relation. Mais le langage est complexe :

```
"Le chat que j'ai adopté mange"

- Relation syntaxique : "mange" → "chat" (sujet)
- Relation référentielle : "j'" → locuteur
- Relation temporelle : "ai adopté" → passé
```

Chaque tête peut se spécialiser sur un aspect différent.

---

### Comment ça marche ?

On fait **n_heads** attentions en parallèle, chacune sur une portion de `d_model` :

```
d_model = 384
n_heads = 6
d_k = d_model / n_heads = 64  (par tête)
```

```
                    Embedding (n, 384)
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
    Head 1 (64)        Head 2 (64)   ...  Head 6 (64)
        ↓                  ↓                  ↓
    Attention          Attention         Attention
        ↓                  ↓                  ↓
    Output (64)        Output (64)  ...  Output (64)
        └──────────────────┼──────────────────┘
                           ↓
                    Concat (n, 384)
                           ↓
                    Projection W_O
                           ↓
                    Output (n, 384)
```

---

### Les projections

Chaque tête a ses propres matrices W_Q, W_K, W_V :

```python
# Pour chaque tête i
Q_i = X @ W_Q_i  # (n, d_model) @ (d_model, d_k) → (n, d_k)
K_i = X @ W_K_i
V_i = X @ W_V_i

head_i = Attention(Q_i, K_i, V_i)  # (n, d_k)
```

En pratique, on fait tout en une seule opération matricielle pour l'efficacité.

---

### Concat + Projection finale

```python
# Concaténer toutes les têtes
concat = [head_1, head_2, ..., head_n]  # (n, n_heads * d_k) = (n, d_model)

# Projection de sortie
output = concat @ W_O  # (n, d_model) @ (d_model, d_model) → (n, d_model)
```

W_O permet de mélanger les informations des différentes têtes.

---

### Exemple concret

```
Modèle : d_model=384, n_heads=6

Entrée: (batch=32, seq=128, d_model=384)

Pour chaque tête (6 fois en parallèle):
  Q, K, V: (32, 128, 64)
  scores:  (32, 128, 128)
  output:  (32, 128, 64)

Concat: (32, 128, 384)
Après W_O: (32, 128, 384)
```

---

### Visualisation des têtes

Après entraînement, on peut visualiser ce que chaque tête "regarde" :

```
Phrase: "Le chat dort sur le canapé"

Tête 1: "dort" regarde fortement "chat"  (sujet-verbe)
Tête 2: "le" regarde "canapé"            (déterminant-nom)
Tête 3: tous regardent les voisins       (localité)
```

---

### Paramètres

```
Par tête:
  W_Q: d_model × d_k
  W_K: d_model × d_k
  W_V: d_model × d_k

Total pour n_heads:
  3 × n_heads × d_model × d_k = 3 × d_model²

Plus W_O:
  d_model × d_model

Total Multi-Head Attention ≈ 4 × d_model²
```

</details>

### Questions de vérification

1. Si d_model=512 et n_heads=8, quelle est la dimension par tête ?
2. Pourquoi utiliser plusieurs petites têtes plutôt qu'une grande ?
3. À quoi sert la matrice W_O ?

---

## Partie 6 : Positional Encoding (RoPE)

L'attention ne connaît **pas l'ordre des mots**. "Le chat mange la souris" et "La souris mange le chat" produiraient le même résultat sans encodage positionnel. **RoPE** (Rotary Position Embedding) injecte la position de chaque token en **tournant** ses vecteurs Q et K dans l'espace.

```
Position 0 → rotation de 0°
Position 1 → rotation de θ°
Position 2 → rotation de 2θ°
...
```

<details>
<summary><strong>📖 Voir les détails complets sur RoPE</strong></summary>

---

### Le problème

L'attention calcule Q × Kᵀ. C'est un produit scalaire, qui est **invariant à l'ordre** :

```
Tokens: ["chat", "mange"]  →  score = Q_chat · K_mange
Tokens: ["mange", "chat"]  →  score = Q_chat · K_mange  (identique !)
```

Le modèle ne sait pas qui vient avant qui.

---

### Anciennes approches

**Positional Encoding sinusoïdal** (Transformer original) :

```
On additionne un vecteur de position à l'embedding :

embedding_final = embedding + position_vector
```

Problème : la position est "mélangée" avec le sens du mot.

**Positional Embedding appris** (GPT-2) :

```
Table de positions apprise : (max_seq_len, d_model)
embedding_final = embedding + position_embedding[pos]
```

Problème : limité à max_seq_len positions vues à l'entraînement.

---

### RoPE : l'idée

Au lieu d'**ajouter** la position, on **tourne** les vecteurs Q et K.

L'idée clé : deux tokens à la position i et j auront un score d'attention qui dépend uniquement de leur **distance relative** (j - i), pas de leur position absolue.

---

### Comment ça marche ?

On prend les dimensions de Q et K **par paires** et on applique une rotation 2D :

```
Dimensions [0,1] : rotation de pos × θ₁
Dimensions [2,3] : rotation de pos × θ₂
Dimensions [4,5] : rotation de pos × θ₃
...
```

Chaque paire tourne à une fréquence différente :

```
θ_i = 1 / (10000^(2i/d_k))

θ₁ = 1/10000^0     = 1.0       (haute fréquence)
θ₂ = 1/10000^0.031 = 0.90      (...)
...
θ₃₂ = 1/10000^1    = 0.0001    (basse fréquence)
```

---

### Rotation 2D

Pour une paire de dimensions (q₀, q₁) à la position pos :

```
q₀' = q₀ × cos(pos × θ) - q₁ × sin(pos × θ)
q₁' = q₀ × sin(pos × θ) + q₁ × cos(pos × θ)
```

C'est une simple rotation dans le plan.

---

### Pourquoi ça encode la distance relative ?

Quand on calcule Q_i · K_j après rotation :

```
score(i, j) = f(q, k, i-j)
```

Le score ne dépend que de la **différence** (i-j), pas des positions absolues. Le modèle comprend naturellement que :
- "chat" est 2 positions avant "mange"
- Peu importe que ce soit aux positions (0,2) ou (5,7)

---

### Avantages de RoPE

| Propriété | RoPE | Sinusoïdal | Appris |
|-----------|------|------------|--------|
| Distance relative | Oui | Non | Non |
| Extrapolation (seq plus longues) | Bonne | Moyenne | Mauvaise |
| Paramètres supplémentaires | 0 | 0 | max_seq × d |
| Utilisé par | LLaMA, Mistral | Transformer orig. | GPT-2 |

---

### En résumé

```
Q, K (n, d_k)
      ↓
  Rotation par position (RoPE)
      ↓
Q_rot, K_rot (n, d_k)
      ↓
  Attention classique (Q_rot × K_rotᵀ / √d_k)
```

</details>

### Questions de vérification

1. Pourquoi l'attention seule ne connaît pas l'ordre des mots ?
2. Quelle est la différence entre ajouter la position et tourner les vecteurs ?
3. Pourquoi la distance relative est préférable à la position absolue ?

---

## Partie 7 : Feed-Forward, RMSNorm, connexions résiduelles

Après l'attention, chaque token passe dans un **réseau Feed-Forward** qui transforme l'information individuellement. **RMSNorm** stabilise les valeurs, et les **connexions résiduelles** (x + f(x)) permettent au gradient de circuler même dans un réseau très profond.

```
x → RMSNorm → Attention → + x → RMSNorm → Feed-Forward → + x
                          ↑ résiduel                      ↑ résiduel
```

<details>
<summary><strong>📖 Voir les détails complets sur FFN, RMSNorm et résiduel</strong></summary>

---

### Feed-Forward Network (FFN)

Après l'attention (qui mélange les tokens), le FFN traite **chaque token indépendamment** :

```python
def feed_forward(x):          # x: (n, d_model)
    hidden = x @ W1 + b1      # (n, d_model) → (n, d_ff)
    hidden = activation(hidden)
    output = hidden @ W2 + b2  # (n, d_ff) → (n, d_model)
    return output
```

---

### Dimension cachée d_ff

Le FFN projette d'abord vers un espace plus grand, puis revient :

```
d_model (384) → d_ff (1024) → d_model (384)
```

Typiquement : `d_ff ≈ 2.7 × d_model` (avec SwiGLU).

Pourquoi ? L'espace élargi permet des transformations plus riches.

---

### SwiGLU (activation moderne)

Les anciens Transformers utilisaient ReLU. Les modèles modernes (LLaMA, Mistral) utilisent **SwiGLU** :

```python
# ReLU classique
hidden = ReLU(x @ W1)

# SwiGLU (plus performant)
gate = sigmoid(x @ W_gate) * (x @ W_gate)  # "porte"
hidden = gate * (x @ W1)
```

SwiGLU utilise une matrice supplémentaire (W_gate) mais donne de meilleurs résultats.

```
Paramètres FFN:
  Classique: 2 × d_model × d_ff
  SwiGLU:    3 × d_model × d_ff (W1, W2, W_gate)
```

---

### RMSNorm

Normalise les vecteurs pour stabiliser l'entraînement :

```python
def rmsnorm(x):
    rms = sqrt(mean(x²))  # Racine de la moyenne des carrés
    return (x / rms) * gamma  # gamma = paramètre appris
```

Comparaison avec LayerNorm :

| | LayerNorm | RMSNorm |
|---|-----------|---------|
| Centrage (- moyenne) | Oui | Non |
| Normalisation | Oui | Oui |
| Vitesse | Plus lent | Plus rapide |
| Utilisé par | GPT-2, BERT | LLaMA, Mistral |

RMSNorm est plus simple et tout aussi efficace.

---

### Pre-Norm vs Post-Norm

**Post-Norm** (Transformer original) :
```
x → Attention → Add(x) → LayerNorm → FFN → Add → LayerNorm
```

**Pre-Norm** (GPT moderne, LLaMA) :
```
x → RMSNorm → Attention → Add(x) → RMSNorm → FFN → Add(x)
```

Pre-Norm est plus stable à l'entraînement, surtout pour les grands modèles.

---

### Connexions résiduelles

Le `+ x` après chaque sous-couche :

```python
# Sans résiduel
x = attention(x)      # Si le gradient disparaît ici, tout est bloqué

# Avec résiduel
x = x + attention(x)  # Le gradient passe toujours via la "route directe"
```

Pourquoi c'est crucial ?

```
Sans résiduel (6 couches):
  gradient × 0.1 × 0.1 × 0.1 × 0.1 × 0.1 × 0.1 = 0.000001 → disparaît

Avec résiduel:
  le gradient a toujours un chemin direct vers chaque couche
```

---

### Le bloc Transformer complet

En combinant tout :

```python
def transformer_block(x):
    # Sous-couche 1 : Attention
    residual = x
    x = rmsnorm(x)
    x = multi_head_attention(x)
    x = residual + x              # connexion résiduelle

    # Sous-couche 2 : Feed-Forward
    residual = x
    x = rmsnorm(x)
    x = feed_forward_swiglu(x)
    x = residual + x              # connexion résiduelle

    return x
```

```
Entrée (n, 384)
    ↓
┌─ RMSNorm → Multi-Head Attention ─┐
│              ↓                    │
└──────────── Add ←─────────────────┘
    ↓
┌─ RMSNorm → Feed-Forward (SwiGLU) ┐
│              ↓                    │
└──────────── Add ←─────────────────┘
    ↓
Sortie (n, 384)
```

</details>

### Questions de vérification

1. Pourquoi le FFN projette vers d_ff > d_model puis revient ?
2. Quel est l'avantage de RMSNorm sur LayerNorm ?
3. Que se passe-t-il sans connexions résiduelles dans un réseau profond ?

---

## Partie 8 : Architecture GPT complète

On assemble tout. GPT empile **N blocs Transformer identiques**, avec un embedding en entrée et une projection vers le vocabulaire en sortie :

```
Tokens → Embedding → [Bloc Transformer × N] → RMSNorm → LM Head → Probabilités
```

Notre modèle : 6 couches, 6 têtes, d_model=384, vocab_size=8192 → **~10M paramètres**.

<details>
<summary><strong>📖 Voir les détails complets sur l'architecture GPT</strong></summary>

---

### Vue d'ensemble

```
Input IDs (batch, seq)
       ↓
Token Embedding          (vocab_size, d_model)
       ↓
┌─────────────────────┐
│  Transformer Block 1 │
│  ┌─ RMSNorm → MHA ─┐│
│  └─── + résiduel ───┘│
│  ┌─ RMSNorm → FFN ─┐│
│  └─── + résiduel ───┘│
├─────────────────────┤
│  Transformer Block 2 │
│        ...           │
├─────────────────────┤
│  Transformer Block 6 │
│        ...           │
└─────────────────────┘
       ↓
RMSNorm finale
       ↓
LM Head (d_model → vocab_size)
       ↓
Logits (batch, seq, vocab_size)
```

---

### Token Embedding

Convertit les IDs en vecteurs :

```python
self.token_emb = nn.Embedding(vocab_size, d_model)
# (batch, seq) → (batch, seq, d_model)
```

Pas de positional embedding séparé : RoPE est appliqué directement dans l'attention.

---

### Les N blocs Transformer

Chaque bloc est identique (mêmes composants, mais poids différents) :

```python
self.layers = nn.ModuleList([
    TransformerBlock(d_model, n_heads, d_ff)
    for _ in range(n_layers)
])
```

Les couches basses captent des patterns simples (syntaxe, proximité), les couches hautes captent des patterns complexes (sémantique, raisonnement).

---

### RMSNorm finale

Après le dernier bloc, une normalisation finale avant la projection :

```python
self.final_norm = RMSNorm(d_model)
```

---

### LM Head (Language Model Head)

Projette les vecteurs vers le vocabulaire pour obtenir les probabilités du prochain token :

```python
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
# (batch, seq, d_model) → (batch, seq, vocab_size)
```

---

### Weight Tying

Astuce : on **partage les poids** entre l'embedding et le LM Head :

```python
self.lm_head.weight = self.token_emb.weight
```

Pourquoi ?
- L'embedding convertit ID → vecteur
- Le LM Head convertit vecteur → ID
- Ce sont des opérations inverses → partager les poids est logique
- Économise `vocab_size × d_model` paramètres (3M dans notre cas)

---

### Forward pass complet

```python
def forward(self, input_ids):
    # 1. Embedding
    x = self.token_emb(input_ids)        # (B, S) → (B, S, 384)

    # 2. N blocs Transformer
    for layer in self.layers:
        x = layer(x)                      # (B, S, 384) → (B, S, 384)

    # 3. Normalisation finale
    x = self.final_norm(x)               # (B, S, 384)

    # 4. Projection vers le vocabulaire
    logits = self.lm_head(x)             # (B, S, 384) → (B, S, 8192)

    return logits
```

---

### Comptage des paramètres

```
Notre modèle (d_model=384, n_layers=6, n_heads=6, vocab_size=8192):

Token Embedding:     8192 × 384          = 3,145,728

Par bloc Transformer (×6):
  RMSNorm (att):     384                 = 384
  W_Q, W_K, W_V:     3 × 384 × 384      = 442,368
  W_O:                384 × 384          = 147,456
  RMSNorm (ffn):     384                 = 384
  FFN (SwiGLU):      3 × 384 × 1024     = 1,179,648
  Sous-total bloc:                       = 1,770,240

6 blocs:             6 × 1,770,240       = 10,621,440

RMSNorm finale:      384                 = 384
LM Head:             partagé (0 extra)   = 0

TOTAL ≈ 13.8M paramètres
```

---

### Comparaison avec d'autres modèles

| Modèle | Paramètres | Couches | d_model | Têtes |
|--------|-----------|---------|---------|-------|
| **Notre mini-GPT** | ~14M | 6 | 384 | 6 |
| GPT-2 Small | 117M | 12 | 768 | 12 |
| GPT-2 XL | 1.5B | 48 | 1600 | 25 |
| LLaMA 7B | 7B | 32 | 4096 | 32 |
| GPT-4 | ~1.8T (estimé) | ? | ? | ? |

Le principe est identique, seule l'échelle change.

</details>

### Questions de vérification

1. Pourquoi partager les poids entre embedding et LM Head ?
2. Quel est le rôle de la RMSNorm finale ?
3. Pourquoi les couches hautes captent des patterns plus complexes ?

---

## Prochaines parties
- **Partie 9** : Entraînement
- **Partie 10** : Génération de texte et inférence

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
