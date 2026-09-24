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


    def mul(self, other):
        return self.__mul__(other)

    def add(self,other):
        return self.__add__(other)

    def sum(self):
        """
        Fonction pour calculer la somme des composants d'un vecteur vers un seul scalaire
        """
        data = self.data.sum()

        out = BasicTensor(data)

        if self.requires_grad:

            out.requires_grad = True
            out.parents = [self]

            # calcul gradient
            def _backward():
                # On fait une somme donc on doit calculer (a_i)' pour avoir la sensibilité a c
                # (a_i)' = 1 donc on à le vecteur 1_N -> [1, 1, ..., 1]
                # 1_N * grad_c  = grad_c étendu n fois :
                self._accumulate(np.full(self.data.shape,out.grad))

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


