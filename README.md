# DIY-ML

Récréer tout, du tensor au GPT, en construisant chaque brique soi-même.

Ce repo est un projet d'apprentissage : je pars des bases du machine learning (un tableau multidimensionnel, un moteur 
de différentiation automatique) et je monte, étape par étape, jusqu'à un petit modèle de langage génératif, en passant par les MLP et les CNN.

## Le principe

Chaque étape existe parce que la précédente bloque sur un truc. Un MLP ne prend qu'une entrée de taille fixe, 
alors que le texte est séquentiel : d'où les embeddings. Une fenêtre de contexte fixe est est pas top : d'où l'attention. 
Etc jusqu'au GPT.

Les blocs de base sont d'abord écrits à la main pour comprendre ce qu'ils font, puis on s'appuie sur les bibliothèques 
standard quand le but devient d'aller plus loin plutôt que de passer son temps à tout réoptimiser :

- le `BasicNDArray` réimplémente les bases de NumPy (buffer plat, shape, strides, vues) en Python pur ;
- l'autograd (`BasicTensor`) tourne sur NumPy, (pour la vitesse !) ;
- PyTorch sert pour comparer dans les tests.

## Avancement

| #  | Étape                                      | Statut  |
|----|--------------------------------------------|---------|
| 00 | Tensor + autograd                          | Fini    |
| 01 | Module, Linear, initialisation             | Début   |
| 02 | Optimiseurs et boucle d'entraînement       | À faire |
| 03 | MLP sur MNIST ?                            | À faire |
| 04 | Embedding                                  | À faire |
| 05 | MLP sur fenêtre de N tokens                | À faire |
| 06 | CNN basique                                | À faire |
| 07 | Attention une tête + masque causal         | À faire |
| 08 | Bloc complet (multi-head, LN, résiduelles) | À faire |
| 09 | Positions (apprises, puis RoPE)            | À faire |
| 10 | GPT char-level + génération                | À faire |
| 11 | BPE, KV cache, top-k / top-p               | À faire |

Le détail est dans `todo.md`.

## Étape 00 : tensors et autograd

### `BasicNDArray` : un NumPy minimalist

- Un buffer plat qui ne bouge jamais, et des métadonnées (shape, strides, offset) qui permettent de savoir la façon de le lire.
- Des vues sans copie et indexation partielle, `transpose`, `broadcast_to` (avec un stride nul sur l'axe étendu), `reshape` (uniquement sur un tableau contigu).
- Un parcours `flat()` qui suit les strides et pas l'ordre du buffer.
- Un `matmul` basique qui fonctionne sur n'importe quelle vue, transposée ou broadcastée, sans code spécifique avec toute 
la lecture qui passe par un seul endroit.

### `BasicTensor` : autograd

- Un DAG construit pendant le forward : chaque opération crée un noeud et lui attache, la fonction qui propage le gradient.
- Un tri topologique (parcours en profondeur postfix, inversé) pour appeler les backward dans le bon ordre.
- L'accumulation des gradients, qui gère les tensors consommés plusieurs fois (résiduelles, poids partagés).
- `unbroadcast`, qui ramène chaque gradient à la shape de son tensor.
- `detach()` et `no_grad()` pour sortir des valeurs du graphe.

Opérations disponibles : `+ − * / ** @`, négation, `sum` et `mean` sur un ou plusieurs axes, `exp`, `log`, `relu`, `maximum`, 
`max` sur un axe, `reshape`, `flatten`, `squeeze`, `expand_dims`, `transpose`, `swapaxes`, l'indexation 
(avec accumulation sur les indices répétés), le log-sum-exp stable, `softmax` (naïf et stable), `log_softmax`, etc.

### Les points clés :

- Pourquoi `transpose` ne copie rien + pourquoi `.contiguous()` existe dans PyTorch.
- Pourquoi le broadcasting coûte rien en mémoire.
- Pourquoi un tri topologique, pourquoi il faut ajouter le noeud après ses parents.
- Pourquoi le softmax naïf casse sur des logits à 1000, et comment le log-sum-exp le répare.

A comprendre plus profondément :
- Pourquoi le backward est un produit jacobienne-vecteur qui ne construit jamais la jacobienne.

## Exemple

```python
import numpy as np
from src.s00_autograd_and_tensors.autograd import BasicTensor

x = BasicTensor(np.random.randn(8, 4))                      # un batch d'entrées
W = BasicTensor(np.random.randn(4, 3), requires_grad=True)  # des poids
b = BasicTensor(np.zeros(3), requires_grad=True)            # un biais

loss = ((x @ W) + b).relu().mean()
loss.backward()

print(W.grad.shape)  # (4, 3)
print(b.grad.shape)  # (3,) : sommé sur les 8 exemples du batch
```

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) et Python 3.14 ou plus récent.

```bash
uv sync
```

PyTorch est installé avec la version de CUDA déclaré dans `pyproject.toml`. Il ne sert pas au code de l'étape 00 mais est utilisé 
comme référence dans les tests. Il sera utilisé pour entrainer les modèles.

## Tests

```bash
uv run pytest
```

Déjà plus de 300 tests, construit de différentes manières :

- Tests vs NumPy pour le `BasicNDArray` : test si avec les mêmes données on a les mêmes résultats, y compris après transpose et broadcast.
- Tests vs PyTorch pour l'autograd : mêmes chose qu'au dessus.
- Gradcheck  `(f(x+ε) − f(x−ε)) / 2ε`, qui vérifie les dérivées sans les connaître à l'avance.
- Tests qui montrent les problèmes comme le softmax naïf qui renvoie `nan` sur de grands logits, alors que PyTorch marche sur la même entrée.

Disclaimer : comme les tests ne sont pas forcément la partie la plus intéressante à écrire dans le projet, j'ai pris
la décision de déléguer leur écriture à des LLM (Claude pour l'instant). 

## Structure

```
src/
  s00_autograd_and_tensors/
    basic_nd_array.py    # tableau n-d à strides, en Python pur
    autograd.py          # BasicTensor, unbroadcast, no_grad
tests/
  s00_autograd_and_tensors/
    basic_nd_array_test.py
    autograd_test.py
    unbroadcast_test.py
```


## Ressources

- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) et [micrograd](https://github.com/karpathy/micrograd)
- [tinygrad](https://github.com/tinygrad/tinygrad)
- [CS231n](https://cs231n.github.io/) de Stanford
- [Dive into Deep Learning](https://d2l.ai/)
- Diverses recherches internet.
- Wikipedia. Quand c'est compréhensible.
- Claude quand vraiment je galère.
- Diverses vidéo de [3blue1brown](https://www.youtube.com/@3blue1brown) 

## Utilisation des IA génératives dans le cadre de ce projet
Par soucis de transparence, je trouve important de mentionner l'utilisation de l'IA générative dans ce projet.
Comme ce projet répond a un objectif pédagogique, je limite cependant son utilisation.

J'utilise le modèle Claude d'Anthropic pour m'aider lors de ce projet :
- Maintenir les .md (dont ce readme). L'IA fait un draft que j'adapte par la suite.
- Ecriture de la grande majorité des tests PyTest.
- Verifier des codes que j'ai rédigés, trouver des problèmes et suggérer des pistes d'améliorations.
- Discuter des concepts de machine learning, expliquer différement certains concepts.
- Proposer des étapes à faire quand je n'ai pas trop d'idées sur quoi avancer; aider pour la roadmap.

Ce que l'IA ne **fait pas** :
- Ecrire la majorité du code
- Réfléchir à ma place

Le projet pourrait probablement être terminé en moins d'une après midi avec quelques agents qui tournent dessus, cependant,
il n'y aurait aucun interet à faire ça. Comme je m'interdis de déléguer le travail à l'IA, et que je travail sur ce
projet sur mon temps libre, il peut cependant avancer plutot lentement.