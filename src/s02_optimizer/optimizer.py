from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, TYPE_CHECKING
import numpy as np

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
    """
        La descente de gradient stochastique est une descente de gradient 'approximative' qui est plus rapide qu'une
        descente de gradient traditionnelle. Cela permet de faire plus d'etape dans un temps donnée.
        Mieux encore, le bruit introduit permet d'eviter de tomber dans des minimums locaux (ou d'en ressortir).

        Attention : ce SGD est stochastique que si un minibatch est fourni en paramètre. Avec un batch complet, il
        est equivalent à une descente de gradient classique
    """
    def __init__(self, params : Iterable[BasicParameter], learning_rate : float):
        super().__init__(params, learning_rate)

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            param.data -= self.learning_rate * param.grad

class MomentumGradient(Optimizer):
    """
        La descente de gradient avec momentum permet de se "souvenir" des gradients précédents : les velocités sont
        une sorte d'EMA (avec dampening = 0 on a plutot une somme pondéré exponentielle),
        qui permet d'accelerer le gradient quand il est dans la bonne direction. On converge donc plus vite.

    """
    def __init__(self, params : Iterable[BasicParameter], learning_rate : float,
                 momentum : float,
                 dampening : float = 0):
        if momentum == 0.0:
            raise ValueError("Momentum must not be zero. Use SGD instead")
        super().__init__(params, learning_rate)
        self.momentum = momentum
        self.dampening = dampening

        # velocité pour chaque param
        self.velocities = {}

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad

            v = self.velocities.get(param)

            if v is None:
                # init de la velocite pour ce param par son gradient
                v = grad.copy()
            else:
                # mise à jour de la velocite
                v = self.momentum * v + (1 - self.dampening) * grad

            self.velocities[param] = v

            param.data -= self.learning_rate * v

class Adam(Optimizer):
    """
        Adam est un mix entre momentum et la RMSprop, avec une correction de biais par dessus.

        - Le momentum est un EMA sur les gradients qui permet de garder la bonne direction et de lisser un peu le bruit.
        - RMSprop divise le pas par la racine d'une EMA des gradients au carré, ce qui adapte le pas à chaque
        paramètre : un paramètre qui a des petits gradients est accéléré tandis qu'un paramètre avec de gros gradients
        et ralentit.
        - Les deux EMA démarrent à zéro et sont donc sous-estimées au début, ce qui est compensé par la correction
        de biais.

        Adam a l'avantage de bien fonctionner avec ses paramètres par défauts, en général.

    """
    def __init__(self, params : Iterable[BasicParameter],
                 learning_rate : float = 1e-3,
                 beta1 : float = 0.9,
                 beta2 : float= 0.999,
                 eps : float= 1e-8,
                 decay : float = 0
                 ):

        super().__init__(params, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.decay = decay

        self.moments = {}
        self.velocities = {}

        self.t = {}


    def step(self):
        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # decay
            if self.decay != 0:
                grad = grad + self.decay * param.data

            # recuperation des valeurs pour le param
            v = self.velocities.get(param, 0)
            m = self.moments.get(param, 0)
            t = self.t.get(param, 1)

            # calcul de m et v
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # correction du biais
            m_corrige = m / ( 1 - (self.beta1 ** t))
            v_corrige = v / (1 - (self.beta2 ** t))

            # mise à jour des valeurs pour le param
            self.moments[param] = m
            self.velocities[param] = v
            self.t[param] = t + 1

            param.data -= self.learning_rate * (m_corrige / (v_corrige ** 0.5 + self.eps))


class AdamW(Adam):

    def __init__(self, params : Iterable[BasicParameter],
                 learning_rate : float = 1e-3,
                 beta1 : float = 0.9,
                 beta2 : float= 0.999,
                 eps : float= 1e-8,
                 decay : float = 1e-2
                 ):
        super().__init__(params, learning_rate, beta1, beta2, eps, decay)

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # recup de valeurs pour le param
            v = self.velocities.get(param, 0)
            m = self.moments.get(param, 0)
            t = self.t.get(param, 1)

            # weight decay decouplé du gradient
            param.data -= self.learning_rate * self.decay * param.data

            # calcul de m et v
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # correction du biais
            m_corrige = m / ( 1 - (self.beta1 ** t))
            v_corrige = v / (1 - (self.beta2 ** t))

            # mise à jour des valeurs pour ce param
            self.moments[param] = m
            self.velocities[param] = v
            self.t[param] = t + 1

            param.data -= self.learning_rate * (m_corrige / (v_corrige ** 0.5 + self.eps))

class AdEMAMix(AdamW):
    """
        Variante d'AdamW avec deux moving average a la place d'un seul
    """
    def __init__(self, params : Iterable[BasicParameter],
                 learning_rate : float = 1e-3,
                 beta1 : float = 0.9,
                 beta2 : float= 0.999,
                 eps : float= 1e-8,
                 decay : float = 1e-2,
                 beta3: float = 0.999,
                 alpha: float = 5.0,
                 t_alpha: int | None = None,
                 t_beta: int | None = None,
                 beta_start: float | None =  None,):
        super().__init__(params, learning_rate, beta1, beta2, eps, decay)
        self.beta3 = beta3
        self.slow_moment = {}
        self.alpha = alpha
        self.t_alpha = t_alpha
        self.t_beta = t_beta

        if beta_start is None:
           self.beta_start = self.beta1
        else:
           self.beta_start = beta_start

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # recup de valeurs pour le param
            v = self.velocities.get(param, 0)
            m = self.moments.get(param, 0)
            m2 = self.slow_moment.get(param, 0)

            t = self.t.get(param, 1)

            if self.t_beta is not None:
                beta = self._fbeta3(t, self.beta3, self.beta_start, self.t_beta)
            else:
                beta = self.beta3

            if self.t_alpha is not None:
                alpha = self._falpha(t, self.alpha, self.t_alpha)
            else:
                alpha = self.alpha


            # weight decay decouplé du gradient
            param.data -= self.learning_rate * self.decay * param.data

            # calcul de m et v
            m = self.beta1 * m + (1 - self.beta1) * grad
            m2 = beta * m2 + (1 - beta) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # correction du biais
            m_corrige = m / ( 1 - (self.beta1 ** t))
            v_corrige = v / (1 - (self.beta2 ** t))

            # mise à jour des valeurs pour ce param
            self.moments[param] = m
            self.slow_moment[param] = m2
            self.velocities[param] = v
            self.t[param] = t + 1

            param.data -= self.learning_rate * ((m_corrige + alpha * m2) / (v_corrige ** 0.5 + self.eps))

    def _falpha(self, t, alpha, t_alpha):
        """
         Scheduler de alpha : montée linéaire de 0 à alpha sur t_alpha pas puis constant.

        :param t: nombre d'etape
        :param alpha: la valeur finale d'alpha
        :param t_alpha: le nombre d'etape de pour avoir alpha a sa valeur finale
        :return: la valeur de alpha pour l'etape t
        """
        return min((t*alpha) / t_alpha, alpha)

    def _fbeta3(self, t, beta3, beta_start, t_beta):
        """
        Scheduler de beta3 : monte de beta_start à beta3 sur t_beta step, puis constant

        :param t: nombre de pas courrant
        :param beta3: la valeur finale de beta 3
        :param beta_start: valeur de départ de beta3
        :param t_beta: nombre d'etape pour avoir beta3 a sa valeur finale
        :return: la valeur de beta3 pour l'etape t
        """

        return min(np.exp(np.log (beta_start) * np.log(beta3)
                          / ((1 - (t / t_beta)) * np.log(beta3)
                            + (t/t_beta) * np.log(beta_start))
                          ), beta3)






