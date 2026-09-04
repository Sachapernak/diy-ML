from __future__ import annotations

import itertools
import math
from typing import Callable

from PIL.ImageChops import offset


class BasicNDArray:
    _array: list[float]
    _strides: list[int]
    _shape: list[int]
    _offset: int

    # ================
    # === Construction
    # ================

    def __init__(self, array: list[float | list] | float):
        self._offset = 0

        # construit un scalar
        if not isinstance(array, list):
            self._array = [array]
            self._strides = []
            self._shape = []
        # construit une list
        else:
            self._shape = BasicNDArray._shape_from_initial_array(array)
            self._strides = BasicNDArray._get_stride(self._shape)
            self._array = self._flatten(array, self._shape)

    def _view(self, stride, shape, offset):
        """
        Construit un vue sur le BasicNDArray courant
        """
        viewed = object.__new__(BasicNDArray)

        viewed._array = self._array

        viewed._strides = stride
        viewed._shape = shape
        viewed._offset = offset

        return viewed

    @staticmethod
    def _from_flat_array(array, shape):
        """
        Construit un nouveau BasicNDArray à partir d'un array flat
        """
        new_object = object.__new__(BasicNDArray)

        new_object._array = array

        new_object._shape = shape
        new_object._strides = new_object._get_stride(shape)

        new_object._offset = 0

        return new_object

    @staticmethod
    def _shape_from_initial_array(current_array) -> list[int]:
        """
        Calcul la Shape a partir de l'array initial
        """

        buffer = []
        current = current_array

        while isinstance(current, list):
            if len(current) != 0:
                buffer.append(len(current))
                current = current[0]
            else:
                buffer.append(0)
                break
        return buffer

    @staticmethod
    def _get_stride(shape: list[int]) -> list[int]:
        """
        Calcul la stride initiale de l'array
        """
        buff_len = len(shape)

        if buff_len == 0:
            return []

        reversed_stride = [1]
        for i in reversed(range(1, buff_len)):
            reversed_stride.append(shape[i] * reversed_stride[-1])

        stride = list(reversed(reversed_stride))
        return stride

    @staticmethod
    def _flatten(array, shape):
        """
        Transforme une list de list en une seule list plate
        """
        buffer = []

        if len(array) != shape[0]:
            raise ValueError("The shape of the array must be consistent")

        if len(array) == 0:
            return buffer

        leaf_seen = False
        branch_seen = False
        for element in array:
            if isinstance(element, list):
                buffer.extend(BasicNDArray._flatten(element, shape[1:]))
                branch_seen = True
            else:
                buffer.append(element)
                leaf_seen = True

            if leaf_seen and branch_seen:
                raise ValueError("The shape of the array must be consistent")

        return buffer

    # ================
    # === built in
    # ================

    def __len__(self) -> int:
        self._assert_not_scalar("Length of a scalar is not defined")

        return self.shape[0]


    def __getitem__(self, item: int | tuple[int, ...]) -> BasicNDArray | float:

        # Transforme en tuple si besoin
        if isinstance(item, int):
            item = (item,)

        i_len = len(item)

        # Trop de dimensions
        if i_len > len(self._shape):
            raise IndexError(
                f"Too much indexes for shape {self.shape}. Expected at most {len(self.shape)} but got {i_len}")

        offset = self._offset

        # Calcul la stride et la shape du nouvel objet
        for pos, i in enumerate(item):
            original_i = i

            if i < 0:
                i = self.shape[pos] + i

            if i >= self._shape[pos] or i < 0:
                raise IndexError(f"Index out of range at position {pos}, value {original_i}. Shape is {self.shape}")

            offset += self._strides[pos] * i

        if i_len == len(self._shape):
            # Scalar
            return self._array[offset]
        else:
            # BasicNDArray
            return self._view(stride=self._strides[i_len:], shape=self._shape[i_len:], offset=offset)

    def __iter__(self):
        self._assert_not_scalar("Cannot iterate over a scalar")

        for i in range(self.shape[0]):
            yield self[i]

    def __repr__(self):
        if self.is_scalar:
            return repr(self[()])

        repr_list = []

        # Calcul récursivement le string des éléments du BasicNDArray
        for element in self:
            repr_list.append(repr(element))

        res = ", ".join(repr_list)

        return "[" + res + "]"

    # ================
    # === Outils
    # ================

    def flat(self):
        """
        Yield les valeurs du tableau dans l'ordre row-major
        """
        ndim = self.nb_dim

        if len(self._array) == 0:
            return

        # Scalaire, rien a parcourir.
        if ndim == 0:
            yield self._array[self._offset]
            return

        # axe de taille 0 -> tableau vide
        if any(s == 0 for s in self._shape):
            return

        # indices + position de l'elem courant
        counter = [0] * ndim
        offset = self._offset

        while True:
            yield self._array[offset]

            # nb d'axes dans le ndarray
            i = ndim - 1

            # tant qu'il reste des axes a incrémenter
            while i >= 0:
                # Check si il reste des elements dans l'axe courant
                if counter[i] < self._shape[i] - 1:
                    counter[i] += 1
                    offset += self._strides[i]
                    break

                # On a parcouru tout l'axe
                else:
                    # Reset de l'axe + on passe au prochain axe
                    offset -= (self._shape[i] - 1) * self._strides[i]
                    counter[i] = 0
                    i -= 1

            if i < 0:
                return

    def array_equal(self, other, epsilon: float | None = None) -> bool:
        """
        Verifie si deux BasicNDArray sont égaux

        Si un epsilon est renseigné, considère les valeurs égales si leur différence est inférieure
        a epsilon.

        Sinon, utilise math.isclose
        """
        if not isinstance(other, BasicNDArray):
            return False

        if self.shape != other.shape:
            return False

        if epsilon is None:
            if any(not math.isclose(a, b) for a, b in zip(self.flat(), other.flat())):
                return False
        else:
            if any(not abs(a - b) < epsilon for a, b in zip(self.flat(), other.flat())):
                return False

        return True


    # ===========
    # == Maths
    # ===========

    def transpose(self, axes: tuple[int, ...] = ()) -> BasicNDArray:
        """
        Transpose le BasicNDArray.

        Si des axes sont donnés, transpose selon ces axes, sinon fait un transposé de matrice.

        len(axes) doit etre égal a nb_dim et tous les axes doivent apparaitrent exactement une seule fois.
        """

        self._assert_not_scalar("Cannot transpose a scalar")

        if len(axes) == 0:
            # Transpose de matrice, reverse la stride et la shape
            return self._view(stride=self.strides[::-1], shape=self.shape[::-1], offset=self._offset)
        else:
            # Check si on a le bon nombre d'axes
            if len(axes) != self.nb_dim:
                raise ValueError("The number of given axes must be the same as the number of dimensions of the BasicNDArray")

            if len(set(axes)) != len(axes):
                raise ValueError("the given axes must not repeat")

            if sorted(axes) != list(range(self.nb_dim)):
                raise ValueError("The given axes must belong to [0, nb_dim) and must all be given")

            new_shape = []
            new_stride = []

            # réordonne les axes selon les indications données
            for axis in axes:
                new_shape.append(self.shape[axis])
                new_stride.append(self.strides[axis])

            return self._view(stride=new_stride, shape=new_shape, offset=self._offset)

    def broadcast_to(self, shape):
        """
        Etend un BasicNDArray vers une shape plus grande
        """
        nb_dim = self.nb_dim
        shape_len = len(shape)

        if shape_len < nb_dim:
            raise ValueError("The given shape must be of equal to or bigger than nb_dim")

        for i in range(nb_dim):
            if self.shape[-i-1] != 1 and self.shape[-i-1] != shape[-i-1]:
                raise ValueError(f"axis {nb_dim - i - 1} of size {self.shape[-i-1]} cannot be broadcast to {shape[-i-1]}")

        # construit les shape et stride avec padding devant pour comparer avec la shape cible
        padded_shape = [1] * (shape_len - nb_dim)
        padded_strides = [0] * (shape_len - nb_dim)

        padded_shape.extend(self.shape)
        padded_strides.extend(self.strides)

        # stride resultat
        new_stride = [0] * shape_len
        # shape resultat = shape passé en param

        for i in range(shape_len):
            # Toutes les shape qui ne sont pas de taille 1 gardent leur stride précedente, sinon, stride à 0
            # note : stride = 0 revient à répéter la meme valeur jusqu'à atteindre la shape sur l'axe souhaité
            if padded_shape[i] != 1:
                new_stride[i] = padded_strides[i]

        return self._view(stride=new_stride, shape=shape, offset=self._offset)


    # TODO : Ajouter matmul avec vecteurs
    def matmul(self, other):
        if self.nb_dim < 2 or other.nb_dim < 2:
            raise ValueError("Given BasicNDArray must have at least 2 dimensions. Vectors and batchs are not supported (yet).")

        # Check si le nombre de colonne de self est egal au nombre de ligne de other
        if self.shape[1] != other.shape[0]:
            raise ValueError("other.shape[0] must equal self.shape[1]")

        res = []
        shape = [self.shape[0], other.shape[1]]

        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                accumulated = 0
                for k in range(self.shape[1]):
                    accumulated += self[i,k] * other[k,j]
                res.append(accumulated)

        return self._from_flat_array(res, shape)


    def __matmul__(self, other):
        return self.matmul(other)

    def reshape(self, shape):
        """
        Change la shape de l'array vers une autre shape.

        La shape d'arrivée doit contenir autant d'éléments que celle d'origine.
        Cette opération n'est autorisée que sur des array contigües.
        """
        total_len = 1
        for shape_value in shape:
            total_len *= shape_value

        if self.size != total_len:
            raise ValueError("The given shape does not give out the same element count as this BasicNDArray.")

        if not self.is_contiguous:
            raise ValueError("The current BasicNDArray is not a contiguous array.")

        return self._view(stride=self._get_stride(shape), shape=shape, offset=self._offset)


    # ================
    # === Properties
    # ================
    @property
    def is_contiguous(self):
        """
        Vérifie si le tableau est contiguë

        Càd que les strides sont les meme que celles d'un tableau contiguë de meme shape
        """
        flat_strides = self._get_stride(self._shape)
        return list(self._strides) == flat_strides


    @property
    def shape(self) -> tuple[int, ...]:
        """
        Renvoie la forme du BasicNDArray

        :return: la forme du BasicNDArray
        """
        return tuple(self._shape)

    @property
    def strides(self) -> tuple[int, ...]:
        """
        Renvoie la stride du BasicNDArray

        :return: les strides du BasicNDArray
        """
        return tuple(self._strides)

    @property
    def nb_dim(self) -> int:
        return len(self._shape)

    @property
    def is_scalar(self) -> bool:
        return self.nb_dim == 0

    @property
    def size(self) -> int:
        """
        Calcul le nombre d'elements dans le BasicNDArray

        :return: la taille du BasicNDArray
        """
        return math.prod(self.shape)

    @property
    def T(self):
        return self.transpose()


    # ======================
    # === Helpers internes
    # ======================
    def _assert_not_scalar(self, message: str) -> None:
        if self.is_scalar:
            raise TypeError(message)
