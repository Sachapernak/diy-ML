from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    # Import pour le typage
    from src.s01_module_and_init.module import BasicParameter

class Optimizer(ABC):

    def __init__(self, params : Iterable[BasicParameter], learning_rate : float):
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        self.params = list(params)
        self.learning_rate = learning_rate

        if not self.params:
            raise ValueError("params must not be empty")

    @abstractmethod
    def step(self):
        raise NotImplementedError("Method step() not implemented")

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

class SGD(Optimizer):
    def __init__(self, params : Iterable[BasicParameter], learning_rate : float):
        super().__init__(params, learning_rate)

    def step(self):
        for param in self.params:
            param.data -= self.learning_rate * param.grad
