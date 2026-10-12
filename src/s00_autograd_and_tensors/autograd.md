# Tensor

## C'est quoi ?
Un tableau multidimensionnel de valeurs numériques, sur lequel on ajoute un autograd pour calculer l'effet de chaque 
paramètre sur la loss.

## Les champs
- **`data`** : un `np.ndarray` en float32
- **`grad`** : un `np.ndarray` de la même shape que `data`, ou `None`. Pas un `Tensor`, puisqu'on ne calcule pas le 
gradient du gradient.
- **`parents`** : les tensors dont celui-ci est issu (zéro pour une "feuille" de l'arbre, un seul pour une op unaire et 
deux pour une op binaire)
- **`_backward`** : la fonction attachée sur l'op, vaut `None` pour les feuilles
- **`requires_grad`** : permet de savoir ce qui accumule du gradient (ex: poids, activations) et ce qui n'en a pas besoin 
(ex : entrées, cibles)

## `backward` vs `_backward`
- `_backward` est un attribut avec la fonction mise par chaque op. Il ne sait faire qu'une chose : redistribuer le gradient 
de son noeud à ses parents.
- `backward()` est la méthode qu'on appelle sur la loss. Elle construit l'ordre, met le gradient de la racine à 1, 
puis appelle tous les `_backward` dans le bon ordre (grâce a un tri topo).

`backward()` prend qu'un scalaire : "le gradient de la sortie vaut 1" veut dire quelque chose que si on a une valeur.


# L'autograd

## C'est quoi ?
L'autograd calcule automatiquement les dérivées des opérations faites sur des tensors, et donc le gradient de la loss 
par rapport à chaque paramètre. C'est ce gradient qui dit dans quel sens bouger chaque poids pour faire baisser la loss.

### Exemple simple
```py
import torch

# Tensor qui utilise l'autograd
x = torch.tensor(4.0, requires_grad=True)

# On fait un calcul
y = x ** 2

# On calcule le gradient
y.backward()
print(x.grad)  # 8.0, car dy/dx = 2x
```

1. On marque un tensor comme nécessitant le gradient
2. Chaque op ajoute un noeud dans un graphe orienté acyclique (DAG) qui garde en mémoire toutes les opérations
3. `backward` calcule le gradient en remontant le graphe
4. La valeur du gradient s'ajoute dans le `.grad` de chaque tensor


## La chain-rule

Pour `c = a + b`, `c = a * b`, etc., le backward ne calcule pas la dérivée seule : il calcule la dérivée locale 
multipliée par le gradient qui arrive.

```
∂L/∂a  =  ∂L/∂c  ×  ∂c/∂a
```

- `∂L/∂c` : `out.grad`, le gradient qui vient d'après
- `∂c/∂a` : la dérivée locale de l'op, la seule chose que le noeud sait
- le produit : ce qu'on accumule dans `a.grad`

On calcul pas `∂L/∂a` d'un coup : ça se construit en chaîne, chaque noeud multiplie sa contribution locale par ce qui 
vient après. C'est ce qui permet d'avoir les dérivées par rapport à pleins de paramètres en une seule passe.

### Les backward de chaque op (`g` = `out.grad`)

| Op          | Gradient de a (et de b)                  |
|-------------|------------------------------------------|
| `a + b`     | `g`, `g`                                 |
| `a - b`     | `g`, `-g`                                |
| `a * b`     | `g * b`, `g * a`                         |
| `a / b`     | `g / b`, `-g * a / b²`                   |
| `a ** b`    | `g * b * a^(b-1)`, `g * a^b * ln(a)`     |
| `A @ B`     | `g @ Bᵀ`, `Aᵀ @ g`                       |
| `sum`       | `g` sur tous les éléments                |
| `mean`      | `g / n` répliqué                         |
| `exp`       | `g * exp(x)`, soit `g * out`             |
| `log`       | `g / x`                                  |
| `relu`      | `g * (x > 0)`                            |
| `max(axis)` | `g` à la position du max, 0 ailleurs     |
| `reshape`   | `g` remis à la shape d'origine           |
| `transpose` | `g` transposé par la permutation inverse |
| `x[idx]`    | `g` replacé aux indices (`np.add.at`)    |


Les deux opérandes du `+`, du `-` et du `*` passent ensuite par `unbroadcast` (voir plus bas).


## Le graphe

Chaque résultat d'opération est un noeud qui retient :
- sa valeur (`data`)
- son gradient (`grad`), `None` tant que rien n'est arrivé
- ses parents, les tensors dont il est issu
- sa fonction `_backward`
- `requires_grad`, vrai si au moins un parent en a besoin

Le forward prépare le backward. Au moment où une op crée son noeud, elle a les valeurs des entrées prête. Elle définit 
alors une fct qui les utilise et l'attache au noeud. Plus tard, `backward()` appelle cette fonction.

On doit faire attention à pas utiliser `.grad` là où il faut `.data`. Seul `out.grad` est un gradient. Les autres 
valeurs (le `b` de `∂(a*b)/∂a = b`) sont des données. Au moment du backward, `other.grad` vaut généralement `None`.


## L'ordre : tri topologique

Un noeud ne peut renvoyer son gradient que quand tous ses consommateurs lui ont donné leur participation. Sinon il part 
avec un gradient incomplet.

On l'obtient par un parcours en profondeur postfix : on visite tous les parents, puis on ajoute le noeud à la liste. 
La liste a donc les parents avant les enfants, et `backward` la parcourt à l'envers, en partant de la loss.

La seule différence avec un parcours en profondeur classique est l'endroit où on ajoute le noeud : après la boucle sur 
les parents, pas avant. C'est ce qui fait le tri topologique.

Le tensor ne connaît que ses parents, jamais ses enfants. C'est pour ça qu'on passe par un DFS.


## L'accumulation

Quand un tensor est utilisé plusieurs fois, il reçoit plusieurs contributions de gradient qui s'additionnent :

```
a * a                 → a.grad = 2a
(a * b) + (a * c)     → a.grad = b + c
```

D'où le `+=` dans `_accumulate`, jamais `=`. C'est ce qui rend possibles les trucs résiduels et le partage de poids.

Du coup : deux `backward()` successifs additionnent leurs gradients. Il faut remettre les gradients des paramètres à zéro 
entre deux backward (`zero_grad`). Les activations intermédiaires sont recréées à chaque forward, il n'y a rien à nettoyer.


## L'unbroadcast

`x @ W + b`, avec `x @ W` en `(32, 10)` et `b` en `(10,)` : NumPy étend `b` sur les 32 lignes au forward. Chaque `b[j]` 
a donc influencé 32 sorties, et son gradient est la somme des 32 contributions.

```
   shape = (3,4) -----→ shape = (4,)
[[1, 2, 5, 4],            [1+3+5, 2+1+3, 5+4+2, 4+4+6]   soit   [9, 6, 11, 14]
 [3, 1, 4, 4],
 [5, 3, 2, 6]]

   shape = (3,4) -----→ shape = (3, 1)
[[1, 2, 5, 4],            [[1+2+5+4],                    soit   [[12],
 [3, 1, 4, 4],             [3+1+4+4],                            [12],
 [5, 3, 2, 6]]             [5+3+2+6]]                            [16]]
```

On a deux règles pour le broadcasting  :
- axes en trop à gauche : on les somme et ils disparaissent
- une dimension de la cible vaut 1 : on somme sur l'axe en gardant la dimension (`keepdims`)

C'est l'inverse  du stride nul du `broadcast_to`. Et `sum` fait l'inverse d'`unbroadcast` : lui répète le gradient 
d'une valeur agrégée.

Le gradient a toujours exactement la shape de son tensor. C'est l'`assert` dans `_accumulate`, qui trouve l'op pas 
bonne quand un backward est faux.


## `detach` et `no_grad`

On détache  ce qui ne doit pas recevoir de gradient.

- `detach()` renvoie une copie qui n'est pas dans le DAG. Exemple : le max dans le softmax stable est une constante 
- qu'on choisit, il a pas besoin de gradient. Le `sub` reste dans le graphe par contre.
- `with no_grad():` désactive la construction du DAG.


## Gradcheck

La dérivée = limite du taux d'accroissement. On l'approxime en "perturbant" chaque élément d'une entrée, un par un :

```
(f(x + ε) − f(x − ε)) / 2ε
```

La forme centrée a une erreur en `ε²`alors que la forme simple en `ε`.

- `f` est n'importe quel calcul qui renvoie un scalaire (c'est pourquoi on fait un `.sum()` à la fin)
- chaque élément coûte deux forwards complets : c'est lent donc on réserve ça pour les tests
- l'intérêt est de vérifier des dérivées sans avoir à les connaître à l'avance

En float32, la différence finie est bruitée. Il faut une tolérance relative en plus de l'absolue, sinon les gros 
gradients (`exp`, broadcast) échouent au hasard.
- Exemple de bruit : écart relatif de `1e-4` à `1e-3`, aléatoire
- Exemple de bug : un facteur 2, un signe inversé, des zéros, une shape fausse


## Softmax et cross-entropy

### La cross-entropy
```
H(y, p) = − Σ_j  y_j · log(p_j)
```
Avec une cible one-hot, tous les termes sont nuls sauf un : la loss vaut `−log(p_bonne_classe)`. Elle est petite si le 
modèle donne une forte probabilité à la bonne réponse, et énorme s'il se trompe alors qu'il est certains. 
Sur un batch, on prend la moyenne, pour que l'échelle dépende pas de la taille du batch.

Le NN sort des logits (réels quelconques), pas des probabilités. Le softmax en fait des proba : 
`p_j = exp(z_j) / Σ_k exp(z_k)`.

### Problèmes rencontrés
1. **Softmax naïf** : `exp(1000)` déborde, et `inf / inf` donne `nan`.
2. **Softmax stable** : on soustrait le max de chaque ligne. Le softmax change pas par translation, donc le résultat 
ne change pas, mais le plus grand exposant vaut 0 et `exp` ne déborde plus.
3. **Mais** la CE prend ensuite `log(p)`. Du coup, si `p` s'arrondit à 0, `log(0) = −∞`.
4. **log-softmax** : on calcule directement `log(p)` sans passer par `p` :
   ```
   log(p_j) = z_j − logsumexp(z)
   logsumexp(z) = m + log(Σ_k exp(z_k − m))      avec m = max(z)
   ```
   Sur `[0, 1000, 0]`, la bonne classe 0 donne une loss de 1000 : c'est grand, mais pas infini.

### Le gradient : `p − y`
Le gradient de la CE par rapport aux logits vaut :
```
∂L/∂z = (softmax(z) − onehot(y)) / N
```
Il monte la bonne classe et baisse les autres. La version composée y arrive en passant par toutes les ops. 
La version fusionnée l'écrit directement.



