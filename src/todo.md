# Todo

| #  | Bloc                                          | Pourquoi                                                    | Avancement |
|----|-----------------------------------------------|-------------------------------------------------------------|------------|
| 00 | Tensor + autograd                             | Les briques de calcul et le backward                        | FINI       |
| 01 | Module, Linear, init                          | Des modèles avec des paramètres à entraîner                 | En cours   |
| 02 | SGD → momentum → Adam + boucle d'entraînement | Entraîner pour de vrai, premier jalon : XOR qui converge    | -          |
| 03 | MLP sur MNIST                                 | NN de base                                                  | -          |
| 04 | Embedding + bigramme char-level               | Comprendre du texte                                         | -          |
| 05 | MLP sur fenêtre de N tokens                   | Fenêtre de contexte                                         | -          |
| 06 | CNN basique                                   | Autre type de NN (branche latérale, pas requis pour le GPT) | -          |
| 07 | Attention 1 tête + masque causal              | Base des Transformers                                       | -          |
| 08 | Bloc complet (multi-head, LN, résiduelles)    | Morceau de transformer                                      | -          |
| 09 | Positions (apprises → RoPE)                   | Encodage de l'ordre des tokens                              | -          |
| 10 | GPT char-level + génération                   | GPT basique                                                 | -          |
| 11 | BPE, KV cache, top-k/top-p                    | Plus de performances                                        | -          |


## 00 · Tensor + autograd   [OK]

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
- [x] `gradcheck` avec tolérance absolue, relative et message d'erreur
- [x] `cross_entropy(logits, cibles)` construite sur `log_softmax`.
- [x] Pleins de tests

### Optionnel a faire
- [ ] (optionnel) CE fusionnée, backward `(p − y) / N` 
- [ ] (optionnel) BCE

### Limites connues
- `matmul` avec un vecteur 1-D : le forward marche, le backward marche pas (`swapaxes(-1, -2)` sur un tableau 1-D)
- `backward()` seulement sur un scalaire
- Le `BasicNDArray` n'a pas d'arithmétique élément par élément : l'autograd tourne sur NumPy (aussi pour des raisons de performances)


## 01 · Module, Linear, init

### Fait
- [ ] Module basic
### Todo
- [ ] Module `Linear` : `x @ W + b`
- [ ] Init He / Xavier, puis casser volontairement (zéro, trop grande, trop petite) et regarder les activations couche par couche

## 02 · Optimiseurs et boucle d'entraînement

### Fait
- [ ] 
### Todo
- [ ] SGD, momentum, Adam
- [ ] `zero_grad` sur les paramètres
- [ ] Overfit sur 10 exemples 
- [ ] XOR 
- [ ] Vérifier que la loss au premier pas vaut `log(C)`
