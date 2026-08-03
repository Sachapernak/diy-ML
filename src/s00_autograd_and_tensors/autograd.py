from typing import Callable

class BasicNDArray:

    _array: list[float]
    _strides: list[int]
    _shape: list[int]
    _offset: int
    def __init__(self, array: list[float | list] | float):
        self._offset = 0

        self._shape = BasicNDArray._shape_from_initial_array(array)
        self._strides = BasicNDArray._get_stride(self._shape)

        if not isinstance(array, list):
            self._array = [array]
        else:
            self._array = self._flatten(array, self._shape)

    def _view(self, stride, shape, offset):
        viewed = object.__new__(BasicNDArray)

        viewed._array = self._array

        viewed._strides = stride
        viewed._shape = shape
        viewed._offset = offset

        return viewed

    @staticmethod
    def _shape_from_initial_array(current_array) -> list[int]:
        buffer = []
        current = current_array
        while isinstance(current, list):
            buffer.append(len(current))
            current=current[0]
        return buffer

    @staticmethod
    def _get_stride(shape: list[int]) -> list[int]:
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

        buffer = []
        if len(array) != shape[0]:
            raise ValueError("The shape of the array must be consistent")

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

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Calcul la forme du tensor

        :return: la forme du tensor
        """
        return tuple(self._shape)

    @property
    def strides(self) -> tuple[int, ...]:
        """
        Calcul la stride du tensor
        :return:
        """
        return tuple(self._strides)

    def __getitem__(self, item: int | tuple[int, ...]) -> BasicNDArray | float:
        if isinstance(item, int):
            item = (item,)

        i_len = len(item)

        if i_len  > len(self._shape):
            raise IndexError(f"Too much indexes for shape {self.shape}. Expected at most {len(self.shape)} but got {i_len}")

        offset = self._offset


        for pos, i  in enumerate(item):
            original_i = i

            if i < 0:
                i = self.shape[pos] + i

            if i >= self._shape[pos] or i < 0:
                raise IndexError(f"Index out of range at position {pos}, value {original_i}. Shape is {self.shape}")

            offset += self._strides[pos] * i

        if i_len  == len(self._shape):
            return self._array[offset]
        else:
            return self._view(stride=self._strides[i_len:], shape=self._shape[i_len:], offset=offset)


    def __repr__(self):
        raise NotImplementedError("__repr__ is not implemented")

    def transpose(self, axes: tuple[int, ...]) -> BasicNDArray:
        raise NotImplementedError("Transpose is not implemented")



class BasicTensor:
    nd_array: BasicNDArray
    gradient: list[float]
    parents: list[BasicTensor]
    backward: Callable | None
    requires_grad: bool

    # TODO