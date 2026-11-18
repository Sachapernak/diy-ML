
import numpy as np

from src.s00_autograd_and_tensors.autograd import BasicTensor
from src.s01_module_and_init.init import Init
from src.s01_module_and_init.module import BasicParameter, BasicModule


class Linear(BasicModule):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias : bool = True,
                 rng : np.random.Generator | None = None,
                 init = Init.he_):
        """
        Initialise un layer Linéaire qui transforme in_features en out_features

        :param in_features:
            Taille de l'entrée
        :param out_features:
            Taille de la sortie
        :param bias:
            Activer ou non l'ajout de biais
        :param rng:
            Générateur random de numpy. Si None, en genere un
        :init:
            Méthode initialisation du module. Par défaut : He

        """
        super().__init__()

        self.init = init

        # convention Pytorch out puis in
        self.weight = BasicParameter(np.zeros((out_features, in_features)))

        self.bias = None

        if bias:
            self.bias = BasicParameter(np.zeros(out_features))

        self.rng = rng if rng is not None else np.random.default_rng()

        self.reset_parameters()


    def reset_parameters(self):
        self.init(self.weight, self.rng)
        if self.bias is not None:
            Init.zeros_(self.bias)


    def forward(self, x: BasicTensor) -> BasicTensor:
        if self.bias is not None:
            return (x @ self.weight.T) + self.bias
        else:
            return x @ self.weight.T

class ReLU(BasicModule):
    def forward(self, x: BasicTensor) -> BasicTensor:
        return x.relu()

class Sequential(BasicModule):
    def __init__(self, *children: BasicModule):
        super().__init__()

        for count, child in enumerate(children):
            self.add_child(str(count), child)

    def forward(self, x: BasicTensor) -> BasicTensor:
        for child in self.children():
            x = child(x)
        return x

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, index: int | slice) -> BasicModule:
        """
        Exemple d'utilisation :
        seq[0] : la première couche
        seq[-1] : la dernière
        seq[1:3] : nouveau Sequential avec ces couches (les couches sont partagées / les memes objets)
        """
        couches = list(self._children.values())
        if isinstance(index, slice):
            return Sequential(*couches[index])
        return couches[index]