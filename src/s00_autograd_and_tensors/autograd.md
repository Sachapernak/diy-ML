# L'autograd

## C'est quoi ?
L'autograd permet de calculer les dérivées pour les opération de Tensor et permet ainsi
de calculer le gradient

### Exemple simple
```py
import torch

# Tensor qui utilise l'autograd
x = torch.tensor(4.0, requires_grad=True)

# On fait un calcul 
y = x ** 2

print(y)  

# On calcul le gradient
y.backward()
print(x.grad)  
```

1. On marque un tensor comme nécessitant l'autograd 
2. Chaque op ajoute un noeud dans le "Direected Acyclic Graph" qui garde en mémoire toutes les OP
3. Backward calcul le gradient en traversant le graph
4. La valeur du gradiant s'ajoute dans le .grad du tensor


# Tensor
## C'est quoi ?
Un tableau / une matrice multi-dimensionnelle qui contient des valeurs numériques
Utile pour faire les grosses opérations mathématiques utilisées en deep learning

### Données de base pour un tensor simple
- Le tableau multi-dimensionnel (un NDArray)
- Le gradient (de meme format que le tableau de données)
- La liste des "parents" pour le graph
- Une fonction de backward pour le calcul du gradient
- Un flag "require_grad" pour pas calculer un gradient si c'est pas nécessaire


