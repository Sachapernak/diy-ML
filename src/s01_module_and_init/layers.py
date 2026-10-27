from typing import Generator

import numpy as np

from s00_autograd_and_tensors.autograd import BasicTensor
from s01_module_and_init.module import BasicParameter
from src.s01_module_and_init.module import BasicModule


class Linear(BasicModule):
    def __init__(self, in_features: int, out_features: int, bias=True, rng : Generator = None):
        """
        Initialise un layer Linéaire qui transforme in_features en out_features
        :param in_features:
            Taille de l'entrée
        :param out_features:
            Taille de la sortie
        """
        super().__init__()
        self.W = BasicParameter(np.zeros((in_features, out_features)))
        self.rng = rng if rng is not None else np.random.default_rng()