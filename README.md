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

| #  | Étape                                      | Statut   |
|----|--------------------------------------------|----------|
| 00 | Tensor + autograd                          | Fini     |
| 01 | Module, Linear, initialisation             | Fini |
| 02 | Optimiseurs et boucle d'entraînement       | Fini  |
| 03 | MLP sur MNIST                              | Fini     |
| 04 | Embedding                                  | À faire  |
| 05 | MLP sur fenêtre de N tokens                | À faire  |
| 06 | CNN basique                                | En cours|
| 07 | Attention une tête + masque causal         | À faire  |
| 08 | Bloc complet (multi-head, LN, résiduelles) | À faire  |
| 09 | Positions (apprises, puis RoPE)            | À faire  |
| 10 | GPT char-level + génération                | À faire  |
| 11 | BPE, KV cache, top-k / top-p               | À faire  |

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


### Les points clés :

- Pourquoi `transpose` ne copie rien + pourquoi `.contiguous()` existe dans PyTorch.
- Pourquoi le broadcasting coûte rien en mémoire.
- Pourquoi un tri topologique, pourquoi il faut ajouter le noeud après ses parents.
- Pourquoi le softmax naïf casse sur des logits à 1000, et comment le log-sum-exp le répare.

A comprendre plus profondément :
- Pourquoi le backward est un produit jacobienne-vecteur qui ne construit jamais la jacobienne.

## Étape 01 : Module, Linear et layers

### `BasicModule` : équivalent à `nn.Module` de PyTorch

- Un arbre de modules : chaque module peut contenir des sous-modules et des paramètres.
- Enregistrement automatique par `__setattr__` : écrire `self.fc1 = Linear(...)` ou `self.W = BasicParameter(...)`. Réaffecter un attribut met à jour l'enregistrement.
- `parameters()` renvoie tous les paramètres du modèle, récursivement. Un poids partagé entre plusieurs modules n'est renvoyé qu'une fois.
- `children()` et `apply()` parcourent l'arbre, avec la même déduplication pour un module enregistré à plusieurs endroits.
- Accès par chemin : `get_submodule("bloc.fc1")`, `get_subparam("bloc.fc1.W")`.
- `train()` / `eval()` propagent le mode à tous les sous-modules (utile pour le Dropout et la BatchNorm).
- `zero_grad()` remet à zéro les gradients de tous les paramètres.
- `monModule(x)` permet de faire directement `monModule.forward(x)`.

### `BasicParameter`

Une sous-classe de `BasicTensor`, avec `requires_grad=True` par défaut. Permet au module de distinguer un poids d'une activation.

### `Init` : les initialisations

- Une fonction générique, `variance_scaling_`, qui tire les poids avec une variance `scale / n`, où `n` est le fan_in, le fan_out ou leur moyenne. C'est l'équivalent du `VarianceScaling` de TensorFlow et JAX.
- He (`he_`), Xavier (`xavier_`) et LeCun (`lecun_`) qui sont des réglages prédéfinis du variance_scaling_, avec loi normale, uniforme ou normale tronquée.
- `normal_` avec un écart-type fixe (comme ce qu'utilise GPT-2), et `zeros_` pour les biais.
- Toutes modifient le paramètre sur place et gardent son dtype. Les fans suivent la convention de PyTorch (sortie en premier), et marchent aussi pour les convolutions.

### Layers

- `Linear` : poids rangés en `(out, in)` comme PyTorch, forward `x @ W.T + b`, biais optionnel, et initialisation He par défaut (modifiable à la construction).
- Chaque couche initialise ses propres paramètres dans `reset_parameters()`.
- `ReLU`, et `Sequential` qui enchaîne des modules, avec `len()`, l'indexation et le slicing (`model[0]`, `model[:2]`).

### Les points clés :

- Pourquoi un paramètre n'est pas juste "un tensor qui a besoin d'un gradient" : une activation en a besoin aussi et un poids gelé n'en a pas besoin mais reste un paramètre.
- Pourquoi l'enregistrement automatique : avec un enregistrement manuel, un oubli donne un paramètre qui ne s'entraîne jamais mais sans aucune erreur.
- Pourquoi dédoublonner les paramètres par identité et pas par nom : `fc1.W` et `fc2.W` portent le même nom mais sont deux poids différents, alors qu'un poids partagé ne doit être mis à jour qu'une fois.
- Pourquoi on n'initialise pas les poids à zéro : tous les neurones d'une couche resteraient identiques pour toujours (cf. problème de symétrie).
- Pourquoi c'est la variance qui compte et pas la distribution, et d'où viennent les facteurs : `1/n` pour garder la variance stable d'une couche à l'autre, `2/n` pour compenser ReLU qui en coupe la moitié.
- Pourquoi PyTorch range les poids en `(out, in)` : c'est la convention mathématique `y = Wx`, la transposée ne coûte rien et enfin le partage de poids entre l'embedding et la couche de sortie d'un GPT devient direct.

## Étape 02 : optimiseurs et boucle d'entraînement

### `Optimizer`

- Une classe abstraite qui reçoit `model.parameters()` et un learning rate. `zero_grad()` est commun, `step()` est propre à chaque optimiseur.
- La mise à jour modifie `param.data` directement, en NumPy, hors du graphe. Les paramètres sans gradient sont ignorés.

### Les optimiseurs

- `SGD` : `θ -= lr · g`.
- `MomentumGradient` : une vitesse par paramètre, conservée d'un pas à l'autre. Convention PyTorch : `v = μ · v + (1 − dampening) · g`, avec une copie du gradient au premier pas.
- `Adam` : Moyenne exponentielle des gradients (m, le momentum) et de leurs carrés (v, RMSprop) + une correction de biais pour le démarrage, et un compteur de pas par paramètre. Weight decay en L2.
- `AdamW` : Adam avec le weight decay découplé, appliqué directement aux poids plutôt qu'au gradient.
- `AdEMAMix` : AdamW avec un second momentum, lent (m₂, β₃ = 0.999 ou 0.9999), ajouté à la mise à jour avec un poids α.

### XOR

Le premier réseau entraîné de bout en bout avec le moteur : un MLP `2 → 8 → 1` en MSE, ou `2 → 8 → 2` en cross-entropy, entraîné jusqu'à ce que les 4 points soient bien classés.

Sur 25 graines (les mêmes initialisations pour tous les optimiseurs), avec max 1000 pas :

| Loss | Optimiseur           | Réussis | Médiane  |
|------|----------------------|---------|----------|
| MSE  | SGD (lr 0.1)         | 25/25   | 16 pas   |
| MSE  | Momentum 0.9         | 19/25   | 13 pas   |
| MSE  | Adam (lr 1e-3)       | 23/25   | 374 pas  |
| MSE  | Adam (lr 1e-2)       | 24/25   | 44 pas   |
| CE   | SGD (lr 0.1)         | 23/25   | 32 pas   |
| CE   | Momentum 0.9         | 22/25   | 10 pas   |
| CE   | Adam (lr 1e-3)       | 23/25   | 251 pas  |
| CE   | Adam (lr 1e-2)       | 23/25   | 30 pas   |

Aucun optimiseur ne gagne vraiment : le problème est trop petit. En inspectant les neurones cachés, on voit que les échecs viennent de ReLU qui restent toutes actives (ou toutes mortes) sur les 4 points, le réseau devient linéaire, et un modèle linéaire ne peut pas représenter un XOR. Sur une seed qui réussit, on retrouve au contraire des neurones qui font un AND, des OR, ou ne s'activent que sur un seul point.

### Les points clés :

- Pourquoi l'optimiseur met à jour `.data` hors du graphe : la mise à jour ne doit pas être dérivée.
- Pourquoi SGD et la descente de gradient sont le même code : la seule différence est dans les données (un minibatch ou tout le dataset).
- Pourquoi le momentum à lr égal fait des pas environ 10 fois plus grands : avec `μ = 0.9`, la vitesse est une somme de gradients dont les poids totalisent `1 / (1 − μ) = 10`.
- Pourquoi Adam a besoin d'une correction de biais : m et v démarrent à zéro, et v (avec `β₂ = 0.999`) est bien plus sous-estimé que m. Sans correction, le premier pas serait environ 3 fois trop grand.
- Pourquoi Adam marche avec `lr = 1e-3` presque partout : le pas vaut environ `lr · m / √v`, de l'ordre de `lr` quelle que soit l'échelle des gradients.
- Pourquoi AdamW : avec SGD, ajouter `wd · θ` au gradient revient exactement à rétrécir les poids. Avec Adam, ce terme est divisé par `√v`, et chaque paramètre est régularisé différemment selon l'historique de ses gradients.
- Pourquoi AdEMAMix a deux moments : une seule moyenne exponentielle ne peut pas à la fois suivre les gradients récents et garder du poids sur des gradients vieux de milliers de pas. On a ducoup m_1 pour les gradients récents et m_2 pour les gradients plus ancients.

## Étape 03 : MLP sur MNIST

### Les données

- `charger_mnist()` télécharge une seule fois le `.npz` de MNIST puis le relit depuis `data/`.
- `split()` met de côté une validation (10 % par défaut) : les **indices** sont mélangés mais pas les données, pour que chaque image garde son étiquette.
- `BasicDataLoader` : l'équivalent minimal du `DataLoader` de PyTorch. `for x, y in loader:` parcourt les batchs, dans un nouvel ordre à chaque epoch, et `len(loader)` donne le nombre de batchs. Sans mélange pour la validation.
- `cross_entropy` accepte maintenant directement des indices de classe (comme `F.cross_entropy`) : elle va chercher la log-probabilité de la bonne classe avec `log_probas[arange(N), y]`, sans construire de one-hot.

### Le prétraitement est dans le modèle

- `Flatten` : `(B, 28, 28) → (B, 784)`, en gardant l'axe du batch.
- `Normalize(moyenne, ecart_type)` : centre et réduit les pixels, avec des statistiques calculées sur la partie entraînement uniquement.

Le modèle reçoit donc des pixels bruts, et quelqu'un qui l'utilise n'a pas besoin de connaître la normalisation :

```python
Sequential(Flatten(), Normalize(moyenne, ecart_type),
           Linear(784, 256, rng=rng), ReLU(), Linear(256, 10, rng=rng, init=Init.lecun_))
```

### Entraînement et comparaison

`mlp_mnist.py` entraîne chaque combinaison modèle * optimiseur sur 5 seed (même découpage pour toutes), garde pour chaque graine la meilleure époque en validation, et affiche la moyenne et l'écart-type.

Exactitude de validation à la meilleure époque, 5 graines, 9 époques, batch 128 :

| Modèle (paramètres)                  | SGD lr 0.1     | Momentum 0.9 lr 0.01 | AdamW lr 1e-3  | AdEMAMix lr 2e-4   |
|--------------------------------------|----------------|----------------------|----------------|--------------------|
| linéaire 784 → 10 (7 850)            | 91.52 ± 0.18 % | 91.55 ± 0.07 %       | 91.69 ± 0.12 % | **91.92 ± 0.06 %** |
| bottleneck 784 → 8 → 256 → 10 (11 000) | 83.30 ± 7.89 % | 92.06 ± 0.75 %     | 93.73 ± 0.56 % | **94.09 ± 0.24 %** |
| mlp 784 → 256 → 10 (204 000)         | 97.50 ± 0.11 % | 97.51 ± 0.08 %       | 97.67 ± 0.04 % | **97.68 ± 0.09 %** |
| mlp 784 → 512 → 256 → 10 (536 000)   | 97.65 ± 0.09 % | 97.59 ± 0.08 %       | 97.55 ± 0.23 % | **97.66 ± 0.17 %** |
| mlp 784 → 1024 → 10 (814 000)        | 97.67 ± 0.10 % | 97.60 ± 0.11 %       | 97.62 ± 0.09 % | **97.86 ± 0.11 %** |

Le MLP 256 avec AdamW, évalué une seule fois sur le jeu de test : **97,7 %**.

Ce que j'en retiens :

- L'architecture compte bien plus que l'optimiseur : plus de 5 points entre le modèle linéaire et un MLP, quelques dixièmes au plus entre les optimiseurs.
- Les optimiseurs se valent à peu près avec mes hyperparamètres (un seul lr chacun). AdEMAMix est légèrement devant sur le linéaire et le plus gros MLP. La vraie différence est la vitesse : AdamW et AdEMAMix atteignent leur meilleure époque vers 4-6 puis surapprennent, SGD et le momentum n'ont souvent pas fini à 9 époques.
- Le MLP 256 est le meilleur compromis : 4 fois plus de paramètres ne rapportent que ~0,2 point.
- Le bottleneck fait 2 points de mieux que le linéaire avec un nombre de paramètres comparable, mais il est instable avec SGD à lr 0.1.
- Le découpage de la validation pèse autant que ces écarts : le même modèle passe de 97,4 % à 98,1 % selon les 6 000 images mises de côté.

### La démo

**DISCLAIMER** : Ne connaissant pas Tkinter, j'ai fait l'interface a l'aide de l'IA générative. De plus, j'ai utilisé l'IA pour la 
fonction de téléchargement du dataset de MNIST sans avoir a importer TensorFlow, qui est lourd.

`dessin.py` : une fenêtre tkinter où l'on dessine un chiffre à la souris, avec les 10 probabilités du MLP en temps réel. Le dessin est préparé comme les images de MNIST : recadré dans une boîte de 20 × 20, puis centré sur son centre de masse dans une image 28 × 28. Elle montre bien les limites du modèle : un 9 ou un 1 « à l'européenne », absents de MNIST, sont pris pour un 3 ou un 9, avec plus de 80 % de confiance.

### Les points clés :

- Pourquoi la normalisation fait partie du modèle -> un utilisateur doit pouvoir lui donner une image brute sans connaitre les paramètres choisi à l'entrainement
- La loss initiale vaut environ Log(nb classe) car le modèle non entrainé est censé prédire chaque classe de manière uniforme. Cependant, le choix d'init fait un peu varier cette valeur
- Sur de petit batch, la loss d'entrainement peut beaucoup varier (par exemple passer 0.004 à 0.1 entre deux batch). Le problème est qu'une seule image ratée sur 32 suffit a fortement impacter la loss.
- Différence entre learning rate entre SGD et Adam : le pas de SGD vaut environ `lr × grad` alors que celui d'Adam environ `lr`
- Le early stopping : la loss de validation remonte dès que le modèle commence à mémoriser les exemples plutôt que la règle sous-jacente alors que celle d'entraînement continue de baisser. Le modèle fait de l'overfiting et généralise moins bien : il vaut mieux l'arrêter avant.
- MLP qui se trompe avec 87 % de confiance -> la softmax répartit toujours 100 % entre les 10 classes, meme si l'entrée ressemble à rien de connue. Il faudrait
  ajouter une classe "unknown" par exemple, mais l'entrainement est plus compliqué (c'est.


## Étape 06 : CNN (en cours)

J'ai avancé l'étape 06 avant les embeddings vu que c'est la suite logique du MLP et qu'on peut tester sur MNIST

### La convolution dans l'autograd

Deux approche qui font le meme calcul :

- `ultra_naive_conv` : Version très simple mais particulièrement lente de la convolution. Pour chaque position de sortie
, on découpe le patch pixel par pixel, on empile tous les patchs avec `stack`, puis un seul matmul avec le kernel flatten. Ca construit des milliers de nœuds dans le graphe donc c'est surtout pour tester
- `cross_corr2d` : Version optimisée. Au lieu de loop sur les positions de sortie, on boucle sur les cases du kernel: pour chaque case `(u, v)`, toute l'image décalée de `(u, v)` est une slice qu'on multiplie par le poids de la case du kernel. Avec un kernel 3x3 ca fait 9 tranches, au lieu d'une boucle sur toutes les positions de l'image. Cette version gère plusieurs canaux d'entrée et de sortie (noyau `(C_out, C_in, kh, kw)`, avec un `@` sur l'axe des canaux), le biais, le stride et la dilatation.


Autres ajouts :

- `stack` : empile des BasicTensor sur un nouvel axe avec le backward qui fait l'inverse, redécoupe le grad et rend à chaque BasicTensor d'origine.
- `@` avec des vecteurs 1-D. Marchait pas avant, maintenant ca transforme les vecteurs en une matrice 1D `(1, k)` ou `(k, 1)`, applique les formules habituelles, puis revient à la shape d'origine.

### Les points clés :

- Pourquoi on parle de convolution alors qu'on calcule une cross-correlation : la vraie convolution retourne le noyau, mais comme il est appris, ça ne change rien.
- La technique "decaler et additionner" où on calcul ce que couvre chaque poids du kernel
- Un CNN c'est un cas particulier du MLP avec des poids partagés et presque tous nuls (matrice de Toeplitz par blocs). On le calcule jamais comme ça car la majorité de
  la matrice serait vide.
- Pourquoi la convolution est équivariante par translation : décaler l'image décale la sortie d'autant.
- Le kernel a une shape `(C_out, C_in, kh, kw)` : chaque layer de l'image d'entrée a son propre filtre, et on peut decider d'avoir plusieurs layer en sortie.
  (par exemple (R,G,B,IR) -> (Noir et blanc + temperature) donnerait C_in = 4, C_out = 2). kh et kw est simplement la hauteur et largeur du filtre.


## Exemples

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

Et avec les modules de l'étape 01 :

```python
import numpy as np
from src.s00_autograd_and_tensors.autograd import BasicTensor
from src.s01_module_and_init.module import BasicModule
from src.s01_module_and_init.layers import Linear

class MLP(BasicModule):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.fc2 = Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())

model = MLP()
print(len(list(model.parameters())))   # 4 : deux poids + deux biais

x = BasicTensor(np.random.randn(16, 4))
model(x).mean().backward()
print(model.fc1.weight.grad.shape)     # (8, 4)
```

Et l'entraînement, avec l'étape 02 :

```python
import numpy as np
from src.s00_autograd_and_tensors.autograd import BasicTensor
from src.s01_module_and_init.layers import Sequential, Linear, ReLU
from src.s02_optimizer.optimizer import AdamW

rng = np.random.default_rng(0)
model = Sequential(Linear(4, 16, rng=rng), ReLU(), Linear(16, 3, rng=rng))
optim = AdamW(model.parameters(), learning_rate=1e-2)

x = BasicTensor(rng.standard_normal((32, 4)))
cibles = BasicTensor(np.eye(3)[rng.integers(0, 3, 32)])   # one-hot

for pas in range(100):
    loss = model(x).cross_entropy(cibles)
    optim.zero_grad()
    loss.backward()
    optim.step()
```

Et MNIST, avec l'étape 03 :

```bash
uv run python -m src.s03_mnist.mlp_mnist   # la comparaison des modèles et des optimiseurs
uv run python -m src.s03_mnist.interface   # la démo : dessiner un chiffre
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

Déjà près de 600 tests, construit de différentes manières :

- Tests vs NumPy pour le `BasicNDArray` : test si avec les mêmes données on a les mêmes résultats, y compris après transpose et broadcast.
- Tests vs PyTorch pour l'autograd : mêmes chose qu'au dessus.
- Gradcheck  `(f(x+ε) − f(x−ε)) / 2ε`, qui vérifie les dérivées sans les connaître à l'avance.
- Tests qui montrent les problèmes comme le softmax naïf qui renvoie `nan` sur de grands logits, alors que PyTorch marche sur la même entrée.
- Tests vs PyTorch pour les modules : mêmes paramètres dans le même ordre que `nn.Module`, mêmes sorties et gradients que `nn.Linear`, et la même trajectoire de loss que PyTorch sur 20 pas de SGD (régression et classification).
- Tests statistiques pour les initialisations : un test de Kolmogorov-Smirnov vérifie que les poids suivent la même loi que ceux de `torch.nn.init`.
- Tests vs `torch.optim` pour les optimiseurs : les mêmes paramètres que PyTorch après chaque pas, en float64, sur des minibatchs différents, avec une couche qui n'a de gradient qu'un pas sur deux, et sans `zero_grad` (les gradients s'accumulent). Plus un test qui montre qu'Adam + L2 et AdamW divergent, alors qu'avec SGD les deux sont identiques.
- Tests vs une implémentation de référence pour AdEMAMix (PyTorch n'en a pas) : l'Algorithm 1 du papier en NumPy, avec le gradient analytique de la fonction de Rosenbrock. La même trajectoire avec et sans schedulers, AdamW retrouvé quand α = 0, et une demi-vie de m₂ qui croît linéairement.
- Tests du chargement des données : le découpage et les batchs ne perdent ni ne dupliquent aucun exemple, chaque image garde son étiquette, chaque époque donne un nouvel ordre reproductible. Et le modèle MNIST complet (`Flatten`, `Normalize`, `Linear`, cross-entropy avec des indices) comparé à PyTorch : même loss, mêmes gradients.
- Tests de la convolution : `cross_corr2d` comparée à la version naive (plusieurs canaux, stride, dilatation, noyau rectangulaire, avec et sans biais), au gradcheck, et à `F.conv2d` de PyTorch. Plus un test d'équivariance : décaler l'image décale la sortie.

Disclaimer : comme les tests ne sont pas forcément la partie la plus intéressante à écrire dans le projet, j'ai pris
la décision de déléguer leur écriture à des LLM (Claude pour l'instant). 

## Structure

```
src/
  s00_autograd_and_tensors/
    basic_nd_array.py    # tableau n-d à strides, en Python pur
    autograd.py          # BasicTensor, unbroadcast, no_grad, default_dtype, convolution
  s01_module_and_init/
    module.py            # BasicModule, BasicParameter
    init.py              # Initialisations (He, Xavier, LeCun...)
    layers.py            # Modules spécifiques (Linear, ReLU, Sequential, Flatten, Normalize)
  s02_optimizer/
    optimizer.py         # Optimizer, SGD, MomentumGradient, Adam, AdamW, AdEMAMix
    xor.py               # XOR : comparaison des optimiseurs, inspection des neurones
  s03_mnist/
    donnees.py           # Téléchargement et chargement de MNIST 
    utils.py             # split, minibatch, BasicDataLoader
    mlp_mnist.py         # Modèles, entraînement, évaluation, comparaison
    interface.py         # Démo : dessiner un chiffre, prédiction en temps réel
    resultats_mlp.md     # Résultats du test
tests/
  s00_autograd_and_tensors/
    basic_nd_array_test.py
    autograd_test.py
    unbroadcast_test.py
  s01_module_and_init/
    init_test.py
    module_test.py
  s02_optimizer/
    optimizer_test.py
  s03_mnist/
    mlp_test.py
```


## Ressources

- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) et [micrograd](https://github.com/karpathy/micrograd)
- [tinygrad](https://github.com/tinygrad/tinygrad)
- [CS231n](https://cs231n.github.io/) de Stanford
- [Dive into Deep Learning](https://d2l.ai/)
- [On weight initialization in deep neural networks](https://arxiv.org/abs/1704.08863) de Siddharth Krishna Kumar pour l'init
- [La documentation PyTorch](https://docs.pytorch.org/docs/2.15/pytorch-api.html)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) de Diederik P. Kingma et Jimmy Ba
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) de Ilya Loshchilov et Frank Hutter
- [The AdEMAMix Optimizer: Better, Faster, Older](https://arxiv.org/abs/2409.03137v2) de Matteo Pagliardini, Pierre Ablin et David Grangier
- Diverses recherches internet.
- Wikipedia. Quand c'est compréhensible.
- Claude quand vraiment je galère.
- Diverses vidéo de [3blue1brown](https://www.youtube.com/@3blue1brown) 

## Utilisation des IA génératives dans le cadre de ce projet
Par souci de transparence, je trouve important de mentionner l'utilisation de l'IA générative dans ce projet.
Comme ce projet répond a un objectif pédagogique, je limite cependant son utilisation.

J'utilise le modèle Claude d'Anthropic pour m'aider lors de ce projet :
- Maintenir les .md (dont ce readme). L'IA fait un draft que j'adapte par la suite.
- Ecriture de la grande majorité des tests PyTest.
- Verifier des codes que j'ai rédigés, trouver des problèmes et suggérer des pistes d'améliorations.
- Discuter des concepts de machine learning, expliquer différement certains concepts.
- M'expliquer des maths à 2h du mat' quand j'y comprends plus grand-chose.
- Proposer des étapes à faire quand je n'ai pas trop d'idées sur quoi avancer; aider pour la roadmap.

Ce que l'IA ne **fait pas** :
- Ecrire la majorité du code
- Réfléchir à ma place

Le projet pourrait probablement être terminé en moins d'une après midi avec quelques agents qui tournent dessus, cependant,
il n'y aurait aucun intérêt à faire ça. Comme je m'interdis de déléguer le travail à l'IA, et que je travail sur ce
projet sur mon temps libre, il peut cependant avancer plutôt lentement.