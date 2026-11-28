# Todo

| #  | Bloc                                          | Pourquoi                                                    | Avancement |
|----|-----------------------------------------------|-------------------------------------------------------------|------------|
| 00 | Tensor + autograd                             | Les briques de calcul et le backward                        | FINI       |
| 01 | Module, Linear, init                          | Des modèles avec des paramètres à entraîner                 | FINI   |
| 02 | SGD → momentum → Adam + boucle d'entraînement | Entraîner pour de vrai, premier jalon : XOR qui converge    | FINI       |
| 03 | MLP sur MNIST                                 | NN de base                                                  | FINI       |
| 04 | Embedding + bigramme char-level               | Comprendre du texte                                         | -          |
| 05 | MLP sur fenêtre de N tokens                   | Fenêtre de contexte                                         | -          |
| 06 | CNN basique                                   | Autre type de NN (branche latérale, pas requis pour le GPT) | En cours   |
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
- [x] `stack`, avec un backward qui redistribue le gradient
- [x] `@` avec des vecteurs 1-D (backward corrigé)
- [x] Pleins de tests

### Optionnel a faire
- [ ] (optionnel) CE fusionnée, backward `(p − y) / N` 
- [ ] (optionnel) BCE

### Limites connues
- `backward()` seulement sur un scalaire
- Le `BasicNDArray` n'a pas d'arithmétique élément par élément : l'autograd tourne sur NumPy (aussi pour des raisons de performances)


## 01 · Module, Linear, init     [OK]

### Fait
- [x] Module basic
- [x] Module `Linear` : `x @ W.T + b`
- [x] Init He / Xavier
- [x] Tester les init

## 02 · Optimiseurs et boucle d'entraînement [0K]

### Fait
- [x] Classe `Optimizer` (ABC) : `step()`, `zero_grad()`
- [x] SGD, momentum, Adam 
- [x] XOR en MSE et en CE et comparaison des optimiseurs
- [x] Verification des neurones cachés pour voir pourquoi ça échoue parfois (On a des ReLU mortes ou toujours actives -> le réseau devient linéaire)
- [x] AdamW vs Adam
- [x] Tests de trajectoire vs PyTorch

## 03 · MLP sur MNIST   [OK]

### Fait
- [x] Chargement de MNIST 
- [x] `split` train / validation, `minibatch`, `BasicDataLoader`
- [x] `Flatten` et `Normalize` dans le modèle 
- [x] `cross_entropy` avec des indices de classe
- [x] Loss initiale vérifiée (~ log(10), LeCun sur la dernière couche)
- [x] Boucles d'entraînement et d'évaluation 
- [x] Comparaison 5 modèles * 4 optimiseurs sur 5 graines
- [x] Évaluation unique sur le test : 97,7 % (MLP 256, AdamW)
- [x] Démo interactive : dessiner un chiffre, prédiction en temps réel
- [x] Tests du chargement et du pipeline vs PyTorch

### Plus tard
- [ ] Scheduler du learning rate (cosinus, WSD) : avant le GPT
- [ ] Sauvegarde et chargement des poids (dans `BasicModule`, avec les buffers)
- [ ] Garder les poids de la meilleure époque, et tester ceux-là
- [ ] K-fold cross validation, ou plusieurs graines de découpage
- [ ] Un lr réglé par optimiseur avant de les comparer
- [ ] (optionnel) Augmentation de données, Dropout
- [ ] (optionnel) Afficher les images mal classées

### Limites connues
- Le MLP plafonne vers 98 % : il perd la structure 2D de l'image (→ CNN, étape 06).
- La démo se trompe sur les écritures absentes de MNIST (9 et 1 "européen" vs ), avec une forte confiance.
- Le test a été fait sur les poids de la dernière époque, pas de la meilleure.

## 06 · CNN basique   [En cours]

Fait avant les étapes 04 et 05 car c'est la suite des MLP pour MNIST.

### Fait
- [x] `ultra_naive_conv` : patchs découpés pixel par pixel, empilés avec `stack`, puis un `@`, simple mais super lent
- [x] `cross_corr2d` : version "décaler et additionner", plusieurs canaux d'entrée et de sortie, biais, stride, dilatation
- [x] Tests : comparaison à la version naive, gradcheck, équivariance par translation, comparaison à `F.conv2d`

### À faire
- [ ] Padding : une opération `pad` (backward = la tranche centrale du gradient)
- [ ] Module `Conv2d` : poids `(C_out, C_in, kh, kw)`, biais, `reset_parameters()` avec He (fan_in = C_in × kh × kw)
- [ ] `MaxPool2d` 2 × 2 (`reshape` puis `max` sur les petits axes)
- [ ] Un CNN sur MNIST, comparé au MLP 256
- [ ] Sauvegarde et chargement des poids (un CNN probablement long à réentraîner à chaque lancement de la démo)
- [ ] Démo : option pour choisir entre le MLP et le CNN a la volée
- [ ] (optionnel) Version Toeplitz-block de la convolution pour montrer qu'un CNN c'est un cas particulier du MLP

### Limites connues
- Pas encore de padding
- `cross_corr2d` attend une entrée à 4 axes `(B, C_in, H, W)` : les images MNIST `(B, 28, 28)` doivent passer en `(B, 1, 28, 28)`