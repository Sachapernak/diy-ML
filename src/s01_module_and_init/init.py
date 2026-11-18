from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    # Import pour le typage
    from .module import BasicParameter


Fan = Literal["fan_in", "fan_out", "fan_avg"]
Distribution = Literal["normal", "uniform", "truncated_normal"]

# std d'une loi normale centrée réduite tronquée à +/- 2.
# Valable que à k = 2.
_STD_NORMALE_TRONQUEE_2 = 0.87962566103423978


class Init:
    """
    Fonctions d'initialisation des paramètres.

    Convention des shapes PyTorch, avec sortie en premier.
        - Linear : (out, in)
        - Conv   : (out, in, h, w)
    """

    # ================
    # === Initialisations nommées
    # ================

    @staticmethod
    def he_(param: BasicParameter,
            rng: np.random.Generator,
            distribution: Distribution = "normal") -> None:
        """
        Initialisation de He

        Marche bien avec relu.

        :param param: le paramètre à initialiser
        :param rng: le générateur aléatoire
        :param distribution: "normal", "uniform" ou "truncated_normal"
        """
        Init.variance_scaling_(param, rng, scale=2, fan="fan_in", distribution=distribution)

    @staticmethod
    def xavier_(param: BasicParameter, rng: np.random.Generator,
                distribution: Distribution = "normal") -> None:
        """
        Initialisation de Xavier

        Marche bien avec tanh et sigmoïde.

        :param param: le paramètre à initialiser
        :param rng: le générateur aléatoire
        :param distribution: "normal", "uniform" ou "truncated_normal"
        """
        Init.variance_scaling_(param, rng, scale=1, fan="fan_avg", distribution=distribution)

    @staticmethod
    def lecun_(param: BasicParameter, rng: np.random.Generator,
               distribution: Distribution = "normal") -> None:
        """
        Initialisation de LeCun

        Marche bien avec tanh et SELU.

        :param param: le paramètre à initialiser
        :param rng: le générateur aléatoire
        :param distribution: "normal", "uniform" ou "truncated_normal"
        """
        Init.variance_scaling_(param, rng, scale=1, fan="fan_in", distribution=distribution)

    # ================
    # === Initialisations de base
    # ================

    @staticmethod
    def variance_scaling_(param: BasicParameter, rng: np.random.Generator,
                          scale: float, fan: Fan, distribution: Distribution) -> None:
        """
        Initialisation générique par la variance : Var(w) = scale / n.

        :param param: le paramètre à initialiser
        :param rng: le générateur aléatoire
        :param scale: le numérateur de la variance (2 pour He, 1 pour Xavier et LeCun)
        :param fan: le nombre n utilisé au dénominateur
            - "fan_in"  : nombre d'entrées d'un neurone
            - "fan_out" : nombre de neurones qui reçoivent chaque entrée
            - "fan_avg" : la moyenne des deux
        :param distribution: la loi des poids, toutes réglées sur la même variance
            - "normal"           : normale centrée
            - "uniform"          : uniforme sur [-a, a], avec a = sqrt(3 * var)
            - "truncated_normal" : normale tronquée à ±2 écarts-types
        :raises ValueError: si fan ou distribution est inconnu
        """
        shape = param.shape

        if fan == "fan_in":
            n = Init._fan_in(shape)
        elif fan == "fan_out":
            n = Init._fan_out(shape)
        elif fan == "fan_avg":
            n = (Init._fan_in(shape) + Init._fan_out(shape)) / 2
        else:
            raise ValueError(f"fan inconnu : {fan!r}. Doit être 'fan_in', 'fan_out' ou 'fan_avg'")

        var = scale / n

        if distribution == "normal":
            valeurs = rng.normal(0.0, np.sqrt(var), shape)
        elif distribution == "uniform":
            # Var(U(-a, a)) = a² / 3, donc a = sqrt(3 * var)
            a = np.sqrt(3 * var)
            valeurs = rng.uniform(-a, a, shape)
        elif distribution == "truncated_normal":
            # Couper les queues réduit l'écart-type : on le corrige pour retomber sur var
            std = np.sqrt(var) / _STD_NORMALE_TRONQUEE_2
            valeurs = Init._truncated_normal(rng, std, shape, k=2.0)
        else:
            raise ValueError(f"Distribution inconnue : {distribution!r}. "
                             f"Doit être 'normal', 'uniform' ou 'truncated_normal'")

        param.data = valeurs.astype(param.dtype)

    @staticmethod
    def normal_(param: BasicParameter, rng: np.random.Generator, std: float) -> None:
        """
        Init selon une loi normale centrée, d'écart-type fixe.

        Utilisé par GPT-2 avec std = 0.02

        :param param: le paramètre à initialiser
        :param rng: le générateur aléatoire
        :param std: l'écart-type
        """
        param.data = rng.normal(0.0, std, param.shape).astype(param.dtype)

    @staticmethod
    def zeros_(param: BasicParameter) -> None:
        """
        Init qui met tout à zéro.

        Utile pour init les biais.

        :param param: le paramètre à initialiser
        """
        param.data = np.zeros(param.shape, dtype=param.dtype)

    # ================
    # === Helpers internes
    # ================

    @staticmethod
    def _truncated_normal(rng: np.random.Generator, std: float,
                          size: tuple[int, ...], k: float = 2.0) -> np.ndarray:
        """
        Loi normale centrée d'écart-type std, tronquée à +/- k*std.

        :param rng: le générateur aléatoire
        :param std: l'écart-type de la normale avant troncature
        :param size: la shape du tableau à produire
        :param k: la troncature, en nombre d'écarts-types
        :return: un tableau de shape size
        """
        x = rng.normal(0.0, std, size)
        hors = np.abs(x) > k * std
        while hors.any():
            x[hors] = rng.normal(0.0, std, hors.sum())
            hors = np.abs(x) > k * std
        return x

    @staticmethod
    def _fan_in(shape: tuple[int, ...]) -> int:
        """
        Nombre d'entrées qui arrivent dans un neurone.
        Convention PyTorch : (out, in) pour Linear, (out, in, h, w) pour les conv.

        :param shape: la shape des poids
        :return: in * produit des dimensions du noyau
        :raises ValueError: si la shape a moins de 2 dimensions
        """
        if len(shape) < 2:
            raise ValueError(f"fan_in non défini pour une shape {shape} : il faut au moins 2 dimensions")
        n_sortie, n_entree, *noyau = shape
        return int(n_entree * np.prod(noyau))

    @staticmethod
    def _fan_out(shape: tuple[int, ...]) -> int:
        """
        Nombre de neurones qui recoivent chaque entrée.
        Convention PyTorch : (out, in) pour Linear, (out, in, h, w) pour les conv.

        :param shape: la shape des poids
        :return: out * produit des dimensions du noyau
        :raises ValueError: si la shape a moins de 2 dimensions
        """
        if len(shape) < 2:
            raise ValueError(f"fan_out non défini pour une shape {shape} : il faut au moins 2 dimensions")
        n_sortie, n_entree, *noyau = shape
        return int(n_sortie * np.prod(noyau))