# Todo

| #  | Bloc                                       | Pourquoi                                                     | Avancement      |
|----|--------------------------------------------|--------------------------------------------------------------|-----------------|
| 00 | Tensor + autograd                          | Les briques de calcul et le backward                         | Presque terminé |
| 01 | Module, Linear, init                       | Des modèles avec des paramètres à entraîner                  | -               |
| 02 | SGD → momentum → Adam + boucle d'entraînement | Entraîner pour de vrai, premier jalon : XOR qui converge  | -               |
| 03 | MLP sur MNIST                              | NN de base                                                   | -               |
| 04 | Embedding + bigramme char-level            | Comprendre du texte                                          | -               |
| 05 | MLP sur fenêtre de N tokens                | Fenêtre de contexte                                          | -               |
| 06 | CNN basique                                | Autre type de NN (branche latérale, pas requis pour le GPT)  | -               |
| 07 | Attention 1 tête + masque causal           | Base des Transformers                                        | -               |
| 08 | Bloc complet (multi-head, LN, résiduelles) | Morceau de transformer                                       | -               |
| 09 | Positions (apprises → RoPE)                | Encodage de l'ordre des tokens                               | -               |
| 10 | GPT char-level + génération                | GPT basique                                                  | -               |
| 11 | BPE, KV cache, top-k/top-p                 | Plus de performances                                         | -               |


## 00 · Tensor + autograd

### Fait
- [x] `BasicNDArray` : buffer plat, shape, strides, offset
- [x] Vues sans copie : indexation partielle, `transpose`, `broadcast_to`, `reshape` (si contigu)
- [x] `flat()` en odomètre, `is_contiguous`, `matmul` naïf
- [x] `unbroadcast`, testée vs PyTorch
- [x] `BasicTensor` : graphe, tri topologique post-ordre, accumulation, `_accumulate` avec invariant de shape
- [x] `detach`, `no_grad`
- [x] Ops : `+ - * / ** @`, ops inversées, `neg`, `sum`/`mean` avec axe, `exp`, `log`, `relu`, `maximum`, `max(axis)`
- [x] Ops de forme : `reshape`, `flatten`, `squeeze`, `expand_dims`, `transpose`, `swapaxes`, `__getitem__`
- [x] `lse`, `naive_softmax`, `stable_softmax`, `log_softmax`, `naive_cross_entropy`
- [x] `gradcheck` par différences finies

### Reste à faire
- [ ] `gradcheck` : ajouter une tolérance relative (`rtol`) et un message d'erreur. Les échecs aléatoires sur `exp` et le broadcast viennent de là.
- [ ] Tests PyTorch : cibles en `dtype=torch.long` (sous Windows, `np.random.randint` donne du 32 bits)
- [ ] `cross_entropy(logits, cibles)` construite sur `log_softmax`. La version naïve prend des probabilités et casse sur `log(0)`.
- [ ] Tests des ops ajoutées depuis le dernier fichier de tests :
  - [ ] `__getitem__`, avec l'accumulation sur indices répétés
  - [ ] `transpose`, `swapaxes`, `.T`
  - [ ] `reshape`, `flatten`, `squeeze`, `expand_dims`
  - [ ] `div` et `pow` réécrits en primitives
  - [ ] ops inversées : `2 - x`, `1 / x`, `2 ** x`, `np_array @ x`
  - [ ] `lse`, `stable_softmax`, `log_softmax` (dont le cas `[0, 1000, 0]`)
  - [ ] `detach`, `no_grad`
- [ ] (optionnel) CE fusionnée, backward `(p − y) / N` 
- [ ] (optionnel) BCE

### Limites actuelles
- `matmul` avec un vecteur 1-D : le forward marche, le backward marche pas (`swapaxes(-1, -2)` sur un tableau 1-D)
- `backward()` seulement sur un scalaire
- Le tri topologique est récursif : on peut atteindre la limite de récursion Python (~1000 niveaux) sur un graphe très profond
- `zero_grad` ne remet à zéro que le tenseur lui-même : l'optimiseur devra le faire sur ses paramètres ?
- `stable_softmax` et `lse` calculent le max avec l'op `max` puis le détachent : un nœud inutile dans le graphe (sans effet sur le résultat)
- Le `BasicNDArray` n'a pas d'arithmétique élément par élément : l'autograd tourne sur NumPy (aussi pour des raisons de performances)


## 01 · Module, Linear, init
- [ ] Qui décide quels tenseurs sont des paramètres ? (a voir avant l'optimiseur)
- [ ] `Linear` : `x @ W + b`
- [ ] Init He / Xavier, puis casser volontairement (zéro, trop grande, trop petite) et regarder les activations couche par couche

## 02 · Optimiseurs et boucle d'entraînement
- [ ] SGD, momentum, Adam
- [ ] `zero_grad` sur les paramètres
- [ ] Overfit sur 10 exemples 
- [ ] XOR 
- [ ] Vérifier que la loss au premier pas vaut `log(C)`
