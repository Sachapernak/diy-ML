from __future__ import annotations

from typing import Callable, Iterable

import numpy as np



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


    def __mul__(self, other):
        if not isinstance(other, BasicTensor):
            other = BasicTensor(other)

        # operation: multiplication
        op = lambda a, b: a * b

        data = op(self.data, other.data)

        out = BasicTensor(data)

        if self.requires_grad or other.requires_grad:
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

        if self.requires_grad or other.requires_grad:

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

        if self.requires_grad:

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

        if self.requires_grad or other.requires_grad:

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

        if self.requires_grad or other.requires_grad:

            out.requires_grad = True
            out.parents = [self, other]

            # calcul gradient
            #
            # grad_A = grad @ B.T
            # grad_B = A.T @ grad

            def _backward():

                self._accumulate(unbroadcast(np.matmul(out.grad, other.data.transpose()), self.data.shape))
                other._accumulate(unbroadcast(np.matmul(self.data.transpose(), out.grad), other.data.shape))

            out._backward = _backward

        return out

    def __truediv__(self, other):
        return self.mul(other.pow(-1))

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

        if self.requires_grad:
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
        if not isinstance(other, BasicTensor):
            other = BasicTensor(other)

        op = lambda a, b: np.pow(a, b)

        data = op(self.data, other.data)

        out = BasicTensor(data)

        if self.requires_grad or other.requires_grad:

            out.requires_grad = True
            out.parents = [self, other]

            def _backward():
                # sensibilité de z :
                # a = ba^(b-1)
                # b = a^b * ln(a)
                self._accumulate(unbroadcast(out.grad * other.data * np.pow(self.data, np.subtract(other.data, 1)),
                                             self.data.shape))
                other._accumulate(unbroadcast(out.grad * np.pow(self.data, other.data) * np.log(self.data),
                                              other.data.shape))

            out._backward = _backward

        return out

    def exp(self):
        """
        Exponentielle, element par element.
        """
        data = np.exp(self.data)

        out = BasicTensor(data)

        if self.requires_grad:
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

        if self.requires_grad:

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

        if self.requires_grad or other.requires_grad:

            out.requires_grad = True
            out.parents = [self, other]

            def _backward():
                # sensibilité de z :
                # a = ba^(b-1)
                # b = a^b * ln(a)
                self._accumulate(unbroadcast(out.grad * (self.data > other.data), self.data.shape))
                other._accumulate(unbroadcast(out.grad * (other.data >= self.data), other.data.shape))

            out._backward = _backward

        return out

    def max(self, axis=None, keepdims=False):

        data = np.max(self.data, axis=axis, keepdims=keepdims)

        out = BasicTensor(data)

        if self.requires_grad:
            out.requires_grad = True
            out.parents = [self]

            def _backward():
                if axis is None:
                    # max global, out.data et out.grad sont des scalaires
                    masque = (self.data == out.data)
                    contribution = out.grad * masque
                else:
                    # shape reduite alignée sur la shape d'entree
                    if keepdims:
                        data_alignee = out.data
                        grad_aligne = out.grad
                    else:
                        data_alignee = np.expand_dims(out.data, axis)
                        grad_aligne = np.expand_dims(out.grad, axis)

                    masque = (self.data == data_alignee)
                    contribution = grad_aligne * masque

                self._accumulate(contribution.astype(self.data.dtype))

            out._backward = _backward

        return out



    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            n = self.data.shape[axis]

        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)



    def log(self):
        """
        ln, elem par elem
        """
        data = np.log(self.data)

        out = BasicTensor(data)

        if self.requires_grad:
            out.requires_grad = True
            out.parents = [self]

            def _backward():
                self._accumulate(out.grad / self.data)

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


