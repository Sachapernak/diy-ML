import math

import numpy as np


def split(x, y,
          percent : float = 0.1,
          rng : np.random.Generator | None = None,
          shuffle : bool = True):
    """
    Split un jeu d'entrainement en deux : un jeu d'entrainement et un de validation

    :param x: données
    :param y: labels
    :param rng: generateur de rng numpy
    :param percent: pourcentage (valeur dans (0, 1)) de données dédiée à la validation
    :param shuffle: melange aleatoirement les données
    :return: renvoit x_train, y_train, x_validate, y_validate
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length to stay coherent")

    if percent <= 0 or percent >= 1:
        raise ValueError("Percent must be between 0 and 1, exclusive")

    if rng is None:
        rng = np.random.default_rng()

    ds_len = len(x)
    split_at = int(ds_len * percent)

    if shuffle:
        idx = rng.permutation(ds_len)
    else:
        idx = range(ds_len)

    idx_val = idx[:split_at]
    idx_train = idx[split_at:]

    return x[idx_train], y[idx_train], x[idx_val], y[idx_val],

def minibatch(x,y,
              size : int,
              rng : np.random.Generator | None = None,
              shuffle : bool = True):
    """
    Fait des minibatch dans un ordre aleatoire (sauf si shuffle = false)

    :param x: données
    :param y: labels de x
    :param size: taille du mini-batch
    :param rng: générateur de rng numpy
    :param shuffle: melange aleatoirement les données
    :return: les batchs (x, y). Le dernier batch peut être plus petit.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(x) != len(y):
        raise ValueError("x and y must have the same length to stay coherent")

    if size <= 0:
        raise ValueError("size must be strictly positive")

    ds_len = len(x)

    if shuffle:
        idx = rng.permutation(ds_len)
    else:
        idx = range(ds_len)

    for b_start in range(0, ds_len, size):
        idx_batch = idx[b_start:b_start+size]
        yield x[idx_batch], y[idx_batch]




class BasicDataLoader:
    def __init__(self, x, y,
                 batch_size: int,
                 rng : np.random.Generator | None = None,
                 shuffle : bool = True):

        if len(x) != len(y):
            raise ValueError("x and y must have the same length to stay coherent")

        if batch_size <= 0:
            raise ValueError("size must be strictly positive")

        self.x = x
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size

        if rng is None :
            self.rng = np.random.default_rng()
        else :
            self.rng = rng

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __iter__(self):
        return minibatch(self.x, self.y, self.batch_size, rng=self.rng, shuffle=self.shuffle)

    def dataset_size(self):
        return len(self.x)

    def batch_size(self):
        return self.batch_size







