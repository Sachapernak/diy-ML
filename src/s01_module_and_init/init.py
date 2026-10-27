import numpy as np


class Init:
    @staticmethod
    def _fan_in(shape):
        """
        Nombre d'entrée d'un neuron
        """
        return np.prod(shape[1:])

    @staticmethod
    def zeros_(param):
        # param.data <- des zéros, même shape
        param.data = np.zeros(param.shape).astype(param.dtype)

    @staticmethod
    def uniform_(param, a, b, rng):
        """
        param.data <- distribution uniforme entre a et b
        """
        param.data = rng.uniform(a, b, size=param.shape).astype(param.dtype)


    @staticmethod
    def normal_(param, std, rng):
        # param.data <- tirage normal centré d'écart-type std, même shape
        ...
    @staticmethod
    def he_normal_(param, rng):
        # calcule std à partir de _fan_in(param.data.shape), puis appelle normal_
        ...