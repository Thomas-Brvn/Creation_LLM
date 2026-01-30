# Concepts clés d'un LLM - Récapitulatif

Ce document explique les concepts fondamentaux pour comprendre comment fonctionne un LLM.

---

## 1. Tokenization (BPE)

### C'est quoi ?
Le processus de conversion du texte en nombres que le modèle peut traiter.

### Vocabulaire de base : 256 bytes

| Propriété | Valeur |
|-----------|--------|
| **Nombre** | 256 (fixe, non modifiable) |
| **Origine** | 1 byte = 8 bits = 2⁸ = 256 valeurs possibles |
| **Contenu** | Tous les caractères de base (a-z, A-Z, 0-9, é, è, espace, etc.) |

```
"A" = byte 65
"a" = byte 97
" " = byte 32
"é" = bytes 195 + 169 (UTF-8)
```

**Pourquoi 256 ?** C'est la base de l'informatique, pas un choix. Tout fichier texte est encodé en bytes.

---

### vocab_size (taille du vocabulaire)

| Propriété | Description |
|-----------|-------------|
| **Nom technique** | `vocab_size` |
| **Modifiable ?** | ✅ Oui |
| **Valeur typique** | 4,000 - 100,000 |
| **Notre projet** | 4,000 |

**Formule :**
```
vocab_size = 256 (bytes de base) + 4 (tokens spéciaux) + nombre de merges BPE
```

**Conséquences :**

| vocab_size | Avantages | Inconvénients |
|------------|-----------|---------------|
| **Petit** (1,000-4,000) | Modèle léger, rapide, peu de mémoire | Séquences plus longues, contexte réduit |
| **Moyen** (8,000-32,000) | Bon équilibre | - |
| **Grand** (50,000-100,000) | Séquences courtes, plus de contexte | Modèle lourd, plus de mémoire |

**Exemple :**
```
vocab_size = 4,000  →  "anticonstitutionnellement" = [anti][constitu][tion][nelle][ment] (5 tokens)
vocab_size = 50,000 →  "anticonstitutionnellement" = [anticonstitutionnellement] (1 token)
```

**Valeurs des modèles connus :**

| Modèle | vocab_size |
|--------|------------|
| GPT-2 | 50,257 |
| GPT-4 | ~100,000 |
| LLaMA | 32,000 |
| Notre mini-LLM | 4,000 |

---

### Tokens spéciaux

| Token | ID | Utilité |
|-------|-----|---------|
| `<\|pad\|>` | 0 | Remplissage pour égaliser les longueurs |
| `<\|unk\|>` | 1 | Mot inconnu (jamais utilisé avec BPE) |
| `<\|bos\|>` | 2 | Début de séquence (Beginning Of Sequence) |
| `<\|eos\|>` | 3 | Fin de séquence (End Of Sequence) |

---

## 2. Embeddings

### C'est quoi ?
La transformation d'un ID de token en un **vecteur de nombres** que le modèle peut manipuler.

```
Token "chat" (ID: 742)
        ↓
    Embedding
        ↓
[0.12, -0.45, 0.78, 0.33, ..., 0.56]  ← vecteur de d_model dimensions
```

---

### d_model (dimension du modèle)

| Propriété | Description |
|-----------|-------------|
| **Nom technique** | `d_model` |
| **Modifiable ?** | ✅ Oui |
| **Valeur typique** | 256 - 16,000 |
| **Notre projet** | 384 |

**C'est quoi ?** Le nombre de dimensions du vecteur qui représente chaque token.

**Analogie :** Décrire une couleur
```
1 dimension  → noir ou blanc (256 nuances)
3 dimensions → RGB (16 millions de couleurs)
384 dimensions → nuances infinies pour représenter le "sens" d'un mot
```

**Conséquences :**

| d_model | Avantages | Inconvénients |
|---------|-----------|---------------|
| **Petit** (128-256) | Rapide, léger | Moins de nuances, moins précis |
| **Moyen** (384-768) | Bon équilibre | - |
| **Grand** (1024-16000) | Très expressif, précis | Lent, beaucoup de mémoire |

**Valeurs des modèles connus :**

| Modèle | d_model |
|--------|---------|
| GPT-2 Small | 768 |
| GPT-2 Large | 1,280 |
| GPT-3 | 12,288 |
| GPT-4 | ~16,000+ |
| Notre mini-LLM | 384 |

**Impact sur la mémoire :**
```
Table d'embedding = vocab_size × d_model

Exemple :
4,000 × 384 = 1,536,000 paramètres (~6 MB)
50,000 × 12,288 = 614,400,000 paramètres (~2.5 GB)
```

---

## 3. Contexte (max_seq_len)

### C'est quoi ?
Le nombre maximum de tokens que le modèle peut "voir" en même temps.

| Propriété | Description |
|-----------|-------------|
| **Nom technique** | `max_seq_len` ou `context_length` |
| **Modifiable ?** | ✅ Oui (mais coûteux) |
| **Valeur typique** | 256 - 200,000 |
| **Notre projet** | 256 |

**Exemple :**
```
max_seq_len = 256

[========= 256 tokens maximum =========]

Si la conversation dépasse 256 tokens :
[oublié]["...seuls les 256 derniers tokens sont visibles"]
```

**Conséquences :**

| max_seq_len | Avantages | Inconvénients |
|-------------|-----------|---------------|
| **Petit** (256-512) | Rapide, peu de mémoire | Oublie vite le contexte |
| **Moyen** (2048-8192) | Bon équilibre | - |
| **Grand** (32k-200k) | Mémoire longue | Très lent, énorme mémoire |

**Pourquoi c'est coûteux d'augmenter ?**
```
L'attention coûte O(n²)

256 tokens  → 65,536 opérations
512 tokens  → 262,144 opérations (4x plus)
2048 tokens → 4,194,304 opérations (64x plus)
```

**Valeurs des modèles connus :**

| Modèle | Contexte max | Équivalent |
|--------|--------------|------------|
| GPT-2 | 1,024 tokens | ~2 pages |
| GPT-3 | 4,096 tokens | ~8 pages |
| GPT-4 | 8,192 tokens | ~15 pages |
| GPT-4 Turbo | 128,000 tokens | ~250 pages |
| Claude | 200,000 tokens | ~400 pages |
| Notre mini-LLM | 256 tokens | ~1 paragraphe |

---

## 4. Architecture Transformer

### n_layers (nombre de couches)

| Propriété | Description |
|-----------|-------------|
| **Nom technique** | `n_layers` |
| **Modifiable ?** | ✅ Oui |
| **Valeur typique** | 4 - 96 |
| **Notre projet** | 6 |

**C'est quoi ?** Le nombre de blocs Transformer empilés. Plus de couches = compréhension plus profonde.

**Conséquences :**

| n_layers | Avantages | Inconvénients |
|----------|-----------|---------------|
| **Peu** (4-6) | Rapide | Compréhension superficielle |
| **Moyen** (12-24) | Bon équilibre | - |
| **Beaucoup** (48-96) | Compréhension profonde | Très lent, risque d'instabilité |

---

### n_heads (têtes d'attention)

| Propriété | Description |
|-----------|-------------|
| **Nom technique** | `n_heads` |
| **Modifiable ?** | ✅ Oui |
| **Contrainte** | `d_model` doit être divisible par `n_heads` |
| **Valeur typique** | 4 - 64 |
| **Notre projet** | 6 |

**C'est quoi ?** Le nombre d'attentions parallèles. Chaque "tête" se spécialise sur un aspect différent.

```
Tête 1 : relations sujet-verbe
Tête 2 : relations de proximité
Tête 3 : relations sémantiques
...
```

**Contrainte importante :**
```python
head_dim = d_model // n_heads

# Exemple avec notre config :
384 // 6 = 64  ✓ (OK, divisible)
384 // 5 = 76.8  ✗ (Erreur !)
```

---

### d_ff (dimension feed-forward)

| Propriété | Description |
|-----------|-------------|
| **Nom technique** | `d_ff` |
| **Modifiable ?** | ✅ Oui |
| **Valeur typique** | 4 × d_model |
| **Notre projet** | 1,536 (= 4 × 384) |

**C'est quoi ?** La dimension de la couche cachée dans le réseau feed-forward.

**Règle générale :** `d_ff = 4 × d_model`

---

## 5. Récapitulatif des paramètres

### Notre mini-LLM

| Paramètre | Valeur | Modifiable ? |
|-----------|--------|--------------|
| `vocab_size` | 4,000 | ✅ |
| `d_model` | 384 | ✅ |
| `n_heads` | 6 | ✅ |
| `n_layers` | 6 | ✅ |
| `d_ff` | 1,536 | ✅ |
| `max_seq_len` | 256 | ✅ |
| `dropout` | 0.1 | ✅ |
| bytes de base | 256 | ❌ (fixe) |

### Calcul du nombre de paramètres

```
Embeddings:        vocab_size × d_model
Par couche:        ~4 × d_model² (attention) + ~8 × d_model × d_ff (feed-forward)
Total:             ~10-15M paramètres pour notre config
```

---

## 6. Comment choisir les valeurs ?

### Règles générales

| Ressource | Recommandation |
|-----------|----------------|
| **GPU faible / CPU** | Petit modèle (d_model=256, n_layers=4) |
| **GPU moyen (8GB)** | Modèle moyen (d_model=512, n_layers=8) |
| **GPU puissant (24GB+)** | Grand modèle (d_model=1024+, n_layers=12+) |

### Plus de données = plus grand modèle utile

| Données | Modèle recommandé |
|---------|-------------------|
| ~10 MB | Très petit |
| ~100 MB | Petit |
| ~1 GB | Moyen |
| ~10 GB+ | Grand |

Notre Wikipedia FR (~600MB) → notre config (384, 6 layers) est adaptée.

---

## 7. Glossaire rapide

| Terme | Signification |
|-------|---------------|
| **Token** | Unité de texte (mot, sous-mot, ou caractère) |
| **BPE** | Byte Pair Encoding - algorithme de tokenization |
| **Embedding** | Vecteur représentant le sens d'un token |
| **Attention** | Mécanisme permettant de "regarder" le contexte |
| **Transformer** | Architecture de base des LLMs modernes |
| **Causal** | Ne peut voir que le passé, pas le futur |
| **Feed-Forward** | Réseau de neurones simple (entrée → sortie) |
| **Dropout** | Technique de régularisation (désactive aléatoirement des neurones) |
