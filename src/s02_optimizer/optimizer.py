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









