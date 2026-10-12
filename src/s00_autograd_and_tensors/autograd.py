from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import numpy as np

_GRAD_ENABLED = True

@contextmanager
def no_grad():
    global _GRAD_ENABLED
    previous = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = previous


def unbroadcast(array : np.ndarray, shape) -> np.ndarray:
    """
    Fait l'inverse du broadcast en accumulant les résultats sur les axes restants

    ex:
       shape = (3,4) -----→ shape =(4,)
    [[1, 2, 5, 4],            [1+3+5, 2+1+3, 5+4+2, 4+4+6]    soit   [9, 6, 11, 14]
     [3, 1, 4, 4],
     [5, 3, 2, 6]]

        shape = (3,4) --------→ shape = (3, 1)
    [[1, 2, 5, 4],             [[1+2+5+4],                    soit   [[12],
     [3, 1, 4, 4],              [3+1+4+4],                            [12],
     [5, 3, 2, 6]]              [5+3+2+6]]                            [16]]

    """
    reduced_axis = array.ndim - len(shape)

    if reduced_axis < 0:
        raise ValueError("Shape too big !")

    for i in range(len(shape)):
        if shape[-i - 1] != 1 and shape[-i - 1] != array.shape[-i - 1]:
            raise ValueError(
                f"Given axis {len(shape) - i - 1} of size {shape[-i - 1]} is not compatible with {array.shape[-i - 1]}")

    # np.sum(array, (0,1,2,...)) fait la somme sur chaques axes
    reduced = np.sum(array, axis=tuple(range(reduced_axis)))

    to_reduce = tuple()
    for i in range(len(shape)):
        if shape[i] == 1:
            to_reduce += (i,)

    if len(to_reduce) > 0:
        reduced = np.sum(reduced, axis=to_reduce, keepdims=True)

    if reduced.shape != tuple(shape):
        raise ValueError("Unexpected shape")

    return reduced


class BasicTensor:
    data: np.ndarray
    grad: np.ndarray
    parents: list[BasicTensor]
    requires_grad: bool
    _backward: Callable[[], None] | None

    __array_priority__ = 1000 # pour eviter d'utiliser l'op numpy au lieu de celle du tensor

    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self._backward = None # La fonction de backward
        self.parents = []
        self.grad = None

    def backward(self):
        """
        Fonction pour calculer le gradient

        Compatible uniquement avec les scalaires.

        Fonctionnement :

            1. On calcul l'ordre dans le graph de calcul
            2. On init le gradient à 1.0
            3. Pour chaque element dans l'ordre, on appelle sa fonction backward (si il en a une)
        """

        if self.data.size != 1:
            raise ValueError("Backward only works for scalars")

        order = self._backward_order()

        self.grad = np.ones_like(self.data)

        for t in order:
            if t._backward is not None:
                t._backward()

    def zero_grad(self):
        self.grad = None

    def detach(self):
        return BasicTensor(self.data, requires_grad=False)
    
    @staticmethod
    def _needs_grad(*tensors):
        return _GRAD_ENABLED and any(t.requires_grad for t in tensors)

    # OP avec symbol python
    def __mul__(self, other):
        if not isinstance(other, BasicTensor):
            other = BasicTensor(other)

        # operation: multiplication
        op = lambda a, b: a * b

        data = op(self.data, other.data)

        out = BasicTensor(data)

        if BasicTensor._needs_grad(self, other):
            out.requires_grad = True
            out.parents = [self, other]

            # calcul gradient
            def _backward():
                # Fonction de backward pour mul qui calcul ∂L/∂a  =  ∂L/∂c * ∂c/∂a
                # On a :
                # - ∂L/∂a le gradient du parent,
                # - ∂L/∂c le gradient de out,
                # - ∂c/∂a la dérivée de c par rapport à a,
                # - c = a * b
                #
                # ∂(a*b)/∂a = b -> grad_parent = grad_out * data_other
                self._accumulate(unbroadcast(out.grad * other.data, self.data.shape))
                other._accumulate(unbroadcast(out.grad * self.data, other.data.shape))

            out._backward = _backward

        return out

    def __add__(self, other):
        if not isinstance(other, BasicTensor):
            other = BasicTensor(other)

        # operation : addition
        op = lambda a, b: a + b

        data = op(self.data, other.data)

        out = BasicTensor(data)

        if BasicTensor._needs_grad(self, other):

            out.requires_grad = True
            out.parents = [self, other]

            # calcul gradient
            def _backward():
                # Fonction de backward pour add : on ajoute le gradient du nouveau parent a ses deux parents
                # pour c = a + b le gradient est ∂L/∂a  =  ∂L/∂c * ∂c/∂a
                # avec ∂L/∂a le gradient du parent, ∂L/∂c le gradient de out et ∂c/∂a la dérivée de c par rapport à a
                # La dérivée locale vaut 1 ici, donc on a gradient parent = gradient enfant
                self._accumulate(unbroadcast(out.grad, self.data.shape))
                other._accumulate(unbroadcast(out.grad, other.data.shape))

            out._backward = _backward

        return out

    def __neg__(self):
        data = np.negative(self.data)

        out = BasicTensor(data)

        if BasicTensor._needs_grad(self):

            out.requires_grad = True
            out.parents = [self]

            def _backward():
                self._accumulate(np.negative(out.grad))

            out._backward = _backward

        return out

    def __sub__(self, other):
        if not isinstance(other, BasicTensor):
            other = BasicTensor(other)

        # operation : soustraction
        op = lambda a, b: a - b

        data = op(self.data, other.data)

        out = BasicTensor(data)

        if BasicTensor._needs_grad(self, other):

            out.requires_grad = True
            out.parents = [self, other]

            # calcul gradient
            def _backward():
                # a - b -> = a + (-b), seul b est passé en negatif du coup
                self._accumulate(unbroadcast(out.grad, self.data.shape))
                other._accumulate(unbroadcast(np.negative(out.grad), other.data.shape))

            out._backward = _backward

        return out


    def __matmul__(self, other):
        if not isinstance(other, BasicTensor):
            other = BasicTensor(other)

        op = lambda a, b: np.matmul(a, b)

        data = op(self.data, other.data)

        out = BasicTensor(data)

        if BasicTensor._needs_grad(self, other):

            out.requires_grad = True
            out.parents = [self, other]

            # calcul gradient
            #
            # grad_A = grad @ B.T
            # grad_B = A.T @ grad

            def _backward():

                self._accumulate(unbroadcast(np.matmul(out.grad, np.swapaxes(other.data, -1, -2)),
                                             self.data.shape))
                other._accumulate(unbroadcast(np.matmul(np.swapaxes(self.data, -1, -2), out.grad),
                                              other.data.shape))

            out._backward = _backward

        return out

    def __truediv__(self, other):
        if not isinstance(other, BasicTensor):
            other = BasicTensor(other)

        op = lambda a, b: np.divide(a, b)

        data = op(self.data, other.data)

        out = BasicTensor(data)

        if BasicTensor._needs_grad(self, other):
            out.requires_grad = True
            out.parents = [self, other]

            # calcul gradient
            #
            # grad_A = grad * 1/b
            # grad_B = grad * -a/b²

            def _backward():
                if self.requires_grad:
                    self._accumulate(unbroadcast(out.grad * (1 / other.data), self.data.shape))
                if other.requires_grad:
                    other._accumulate(unbroadcast(out.grad * (-np.divide(self.data, np.power(other.data, 2))), other.data.shape))

            out._backward = _backward
        return out

    def __pow__(self, other):
        if not isinstance(other, BasicTensor):
            other = BasicTensor(other)

        op = lambda a, b: np.power(a, b)

        data = op(self.data, other.data)

        out = BasicTensor(data)

        if BasicTensor._needs_grad(self, other):

            out.requires_grad = True
            out.parents = [self, other]

            def _backward():
                # sensibilité de z :
                # a = ba^(b-1)
                # b = a^b * ln(a)
                if self.requires_grad:
                    self._accumulate(unbroadcast(out.grad * other.data * np.power(self.data, np.subtract(other.data, 1)),
                                             self.data.shape))
                if other.requires_grad:
                    other._accumulate(unbroadcast(out.grad * out.data * np.log(self.data),
                                                  other.data.shape))

            out._backward = _backward

        return out

    # OP inversées
    def __radd__(self, other):
        return self.add(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __rsub__(self, other):
        if not isinstance(other, BasicTensor):
            other = BasicTensor(other)
        return other.sub(self)  # other - self

    def __rtruediv__(self, other):
        if not isinstance(other, BasicTensor):
            other = BasicTensor(other)
        return other.div(self)  # other / self

    def __rmatmul__(self, other):
        if not isinstance(other, BasicTensor):
            other = BasicTensor(other)
        return other.matmul(self)

    def __rpow__(self, other):
        if not isinstance(other, BasicTensor):
            other = BasicTensor(other)
        return other.pow(self)

    # OP de base

    def mul(self, other):
        return self.__mul__(other)

    def add(self,other):
        return self.__add__(other)


    def sum(self, axis=None, keepdims=False):
        """
        Somme le long d'un axe (ou de tous si axis est None).
        """

        data = np.sum(self.data, axis=axis, keepdims=keepdims)

        out = BasicTensor(data)

        if BasicTensor._needs_grad(self):
            out.requires_grad = True
            out.parents = [self]

            def _backward():
                if axis is None:
                    # somme globale
                    grad_aligne = out.grad
                elif keepdims:
                    # shape deja alignee sur celle de l'entree
                    grad_aligne = out.grad
                else:
                    # on remet l'axe pour que le broadcast marche
                    grad_aligne = np.expand_dims(out.grad, axis)

                contribution = np.broadcast_to(grad_aligne, self.data.shape)

                self._accumulate(contribution.astype(self.data.dtype))

            out._backward = _backward

        return out


    def matmul(self, other):
        return self.__matmul__(other)

    def neg(self):
        return self.__neg__()

    def sub(self, other):
        return self.__sub__(other)

    def pow(self, other):
        return self.__pow__(other)

    def exp(self):
        """
        Exponentielle, element par element.
        """
        data = np.exp(self.data)

        out = BasicTensor(data)

        if BasicTensor._needs_grad(self):
            out.requires_grad = True
            out.parents = [self]

            def _backward():
                self._accumulate(out.grad * out.data)

            out._backward = _backward

        return out

    def div(self, other):
        return self.__truediv__(other)

    def relu(self):

        data = np.maximum(0, self.data)

        out = BasicTensor(data)

        if BasicTensor._needs_grad(self):

            out.requires_grad = True
            out.parents = [self]

            def _backward():
                self._accumulate(out.grad * (self.data > 0))

            out._backward = _backward

        return out

    def maximum(self, other):
        if not isinstance(other, BasicTensor):
            other = BasicTensor(other)

        op = lambda a, b: np.maximum(a, b)

        data = op(self.data, other.data)

        out = BasicTensor(data)

        if BasicTensor._needs_grad(self, other):

            out.requires_grad = True
            out.parents = [self, other]

            def _backward():
                self._accumulate(unbroadcast(out.grad * (self.data > other.data), self.data.shape))
                other._accumulate(unbroadcast(out.grad * (other.data >= self.data), other.data.shape))

            out._backward = _backward

        return out

    def max(self, axis=None, keepdims=False):

        data = np.max(self.data, axis=axis, keepdims=keepdims)

        out = BasicTensor(data)

        if BasicTensor._needs_grad(self):
            out.requires_grad = True
            out.parents = [self]

            def _backward():
                if axis is None:
                    # max global, out.data et out.grad sont des scalaires
                    masque = (self.data == out.data)
                    contribution = out.grad * (masque / masque.sum())
                else:
                    # shape reduite alignée sur la shape d'entree
                    if keepdims:
                        data_alignee = out.data
                        grad_aligne = out.grad
                    else:
                        data_alignee = np.expand_dims(out.data, axis)
                        grad_aligne = np.expand_dims(out.grad, axis)

                    masque = (self.data == data_alignee)
                    contribution = grad_aligne * (masque / masque.sum(axis=axis, keepdims=True))

                self._accumulate(contribution.astype(self.data.dtype))

            out._backward = _backward

        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            n = 1
            for a in axes:
                n *= self.data.shape[a]

        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)


    def log(self):
        """
        ln, elem par elem
        """
        data = np.log(self.data)

        out = BasicTensor(data)

        if BasicTensor._needs_grad(self):
            out.requires_grad = True
            out.parents = [self]

            def _backward():
                self._accumulate(out.grad / self.data)

            out._backward = _backward

        return out

    def lse(self, axis=-1, keepdims=False):
        """
        Log Sum Exp stable : log(sum(exp(x))) sur un axe

        Utilisation du log-sum-exp trick pour eviter que ca explose.
        Soustraire le max ne change pas le grad car LSE dépend pas du max choisi
        """
        max_logits = self.max(axis=axis, keepdims=True).detach()

        out = self.sub(max_logits).exp().sum(axis=axis, keepdims=True).log().add(max_logits)

        if not keepdims:
            out = out.squeeze(axis)
        return out

    def naive_softmax(self):
        """
        Fait softmax de version naive. Pose problème avec des logits grands
        -> Préférer stable ou log softmax
        """
        self_exp = self.exp()
        return self_exp.div(self_exp.sum(axis=-1, keepdims=True))


    def stable_softmax(self):
        """
        Version du softmax qui casse pas avec les logits grand.
        Résultat entre [0, 1], mais peut underflow
        -> Utile pour calculer les vraies probabilitiés
        """
        # Recup de la valeur max
        max_logits = self.max(axis=-1, keepdims=True).detach() # detach pour pas calculer le grad

        # Soustraction du max (les vals valent <= 0, donc e^x <= 1)
        exp_logits = self.sub(max_logits).exp()

        # Calcule du softmax normal
        return exp_logits.div(exp_logits.sum(axis=-1, keepdims=True))

    def log_softmax(self, axis=-1):
        """
        Version stable pour logits grand et petit
        Renvoie la log-probabilitée entre (-inf, 0], sans underflow et overflow
        -> Utile pour le calcul de loss
        """
        return self.sub(self.lse(axis=axis, keepdims=True))

    def naive_cross_entropy(self, target):
        # H(P,Q) = - Sum ( log( Q(x) ) * P(x) )
        # P(x) : vrai proba
        # Q(x) : ce qu'on à calculé
        # Pour prendre en compte les batchs on fait mean(H(P,Q))

        return  self.log().mul(target).sum(axis = -1).neg().mean()

    # ================ OP de forme

    def reshape(self, *shape):
        """
        Passe change la shape du tensor vers une autre, tant que c'est compatible
        """
        # Change la shape, tant qu'elle est compatible
        data = self.data.reshape(*shape)
        out = BasicTensor(data)

        if BasicTensor._needs_grad(self):
            out.requires_grad = True
            out.parents = [self]

            def _backward():
                # Meme grad, juste reshape
                self._accumulate(out.grad.reshape(self.data.shape))

            out._backward = _backward

        return out

    def flatten(self, start_axis=1):
        """
        Transforme le tensor en un tableau 1D
        """
        # (B, C, H, W) -> (B, C*H*W) par défaut
        return self.reshape(*self.data.shape[:start_axis], -1)

    def expand_dims(self, axis):
        """
        Ajoute un axe à la pos indiquée en emballant dans une nouvelle dim
        ( [a,b,c] -> [[a,b,c]] par exemple)

        Ex : a = (2, 3)
        - a.expand_dims(0) -> (1, 2, 3)
        - a.expand_dims(-1) -> (2, 3, 1)
        """
        return self.reshape(np.expand_dims(self.data, axis).shape)

    def squeeze(self, axis=None):
        """
        Supprime un axe de taille 1 (tous ou celui donnée en param)
        ([[a,b,c]] -> [a,b,c] par exemple)

        Ex : a = (1, 2, 1, 3)
        - a.squeeze() -> (2, 3)
        - a.squeeze(0) -> (2, 1, 3)
        """
        return self.reshape(np.squeeze(self.data, axis).shape)

    def transpose(self, axes=None):
        """
        Fait la transposée
        """
        if axes is None:
            axes = tuple(reversed(range(self.data.ndim)))

        axes = tuple(a % self.data.ndim for a in axes)

        data = np.transpose(self.data, axes)
        out = BasicTensor(data)

        if BasicTensor._needs_grad(self):
            out.requires_grad = True
            out.parents = [self]

            inverse = np.argsort(axes)

            def _backward():
                self._accumulate(np.transpose(out.grad, inverse))

            out._backward = _backward

        return out

    def swapaxes(self, a1, a2):
        """
        Echange deux axes (change l'ordre des données !)

        Ex : a = (2, 3)
        - a.swapaxes(0, 1) -> (3, 2)
        """
        axes = list(range(self.data.ndim))
        axes[a1], axes[a2] = axes[a2], axes[a1]
        return self.transpose(tuple(axes))

    @property
    def T(self):
        return self.transpose()

    # =============
    # Autres OP
    # =============
    def __getitem__(self, index):
        # si on indexe avec des tensor, on récupère leurs données (en entiers)
        idx = index
        if isinstance(index, BasicTensor):
            idx = index.data.astype(np.int64)
        elif isinstance(index, tuple):
            idx = tuple(i.data.astype(np.int64) if isinstance(i, BasicTensor) else i for i in index)

        data = self.data[idx]
        out = BasicTensor(data)

        if BasicTensor._needs_grad(self):
            out.requires_grad = True
            out.parents = [self]

            def _backward():
                # On fait np.add.at car on doit ajouter au gradient, chaque fois
                # qu'il est utilisé
                # ex :
                #   x = [1., 2., 3., 4., 5.]; idx = [1, 1, 3]; y = x[idx]                # [2., 2., 4.]  -> y[0] et y[1] viennent de x[1], y[2] de x[3]
                #
                #   si par exemple out.grad = [10, 20, 30] :
                #   grad[1] = out.grad[0] + out.grad[1] = 30
                #   grad[3] = out.grad[2]               = 30
                #   grad    = [0, 30, 0, 30, 0]
                grad = np.zeros_like(self.data)
                np.add.at(grad, idx, out.grad)
                self._accumulate(grad)

            out._backward = _backward

        return out


    # =============
    # fonctions internes
    # =============


    def _tri_topo(self, seen, res):
        """
        Tri topologique récursif basé sur un parcours en profondeur "post-order" (on ajoute après les parents)

        1. On check si on a deja vu le courant (exit si oui)
        2. On marque courant comme seen
        3. On check tous ses parents (marque + ajoute)
        4. On ajoute le courant en dernier
        """
        if self not in seen and self.requires_grad:
            seen.add(self)
            for parent in self.parents:
                parent._tri_topo(seen, res)
            res.append(self)


    def _backward_order(self):
        """
        Construit l'ordre de visite pour backward.

        Le Tensor connait ses parents, ce qui permet de créer un graph "DAG" (directed acyclic graph).
        En faisant l'inverse d'un tri topologique, on obtient l'ordre pour backward

        """
        seen = set()
        res = list()

        self._tri_topo(seen, res)

        return res[::-1]


    def _accumulate(self, contribution: np.ndarray) -> None:
        """
        Accumule au gradient du tensor.

        - vérifie que requires grad est a vrai
        - vérifie le format de l'accumulation
        - init le gradient s'il ne l'est pas
        - ajoute la valeur au gradient
        """
        if not self.requires_grad:
            return

        assert contribution.shape == self.data.shape, (
            f"grad shape {contribution.shape} != data shape {self.data.shape}"
        )

        if self.grad is None:
            self.grad = contribution.copy()
        else:
            self.grad += contribution


