"""
MNIST : comparer plusieurs modeles et plusieurs optimiseurs, sur plusieurs graines.

Lancer :  uv run python -m src.s03_mnist.mlp_mnist
"""

import math
from typing import Callable, Any, Iterable

import numpy as np

from src.s01_module_and_init.module import BasicParameter
from src.s00_autograd_and_tensors.autograd import BasicTensor, no_grad
from src.s01_module_and_init.init import Init
from src.s01_module_and_init.layers import Sequential, Flatten, Normalize, Linear, ReLU
from src.s01_module_and_init.module import BasicModule
from src.s02_optimizer.optimizer import SGD, MomentumGradient, AdamW, AdEMAMix, Optimizer
from src.s03_mnist.donnees import charger_mnist
from src.s03_mnist.utils import split, BasicDataLoader


# Le decoupage train / validation est le meme pour tous les runs : seuls l'initialisation
# et l'ordre des batchs changent d'une graine a l'autre.
SEED_SPLIT = 42
VALIDATION_PROP = 0.1

EPOCHS = 9
BATCH_SIZE = 128
NB_SEEDS = 5

DEFAULT_SEED = 0


# ---------------------------------------------------------------------------
# Donnees
# ---------------------------------------------------------------------------

def preparer_donnees(val_fraction: float = 0.1, data_seed = SEED_SPLIT) -> dict:
    """
    Charge MNIST, met de cote une validation, et calcule les statistiques de normalisation
    sur la partie entrainement uniquement.
    """
    x_train, y_train, x_test, y_test = charger_mnist()
    x_train, y_train, x_val, y_val = split(x_train, y_train, percent=val_fraction,
                                           rng=np.random.default_rng(data_seed))
    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
        "moyenne": float(np.mean(x_train)),
        "ecart_type": float(np.std(x_train)),
    }


# ---------------------------------------------------------------------------
# Modeles : fn(moyenne, ecart_type, rng) -> modele
#
# Factory pour pouvoir renseigner moyenne et std plus tard
# ---------------------------------------------------------------------------

def lineaire(moyenne, ecart_type, rng):
    """
    Regression lineaire basique comme baseline
    :param moyenne: moyenne du dataset
    :param ecart_type: ecart-type du dataset
    :param rng: generateur de rng numpy
    :return: modèle de regression linéaire
    """
    return Sequential(
        Flatten(),
        Normalize(moyenne, ecart_type),
        Linear(784, 10, rng=rng, init=Init.lecun_),
    )


def mlp_256(moyenne, ecart_type, rng):
    """
    Petit MLP avec un seul hidden layer 784 -> 256 -> 10
    :param moyenne: moyenne du dataset
    :param ecart_type: ecart type du dataset
    :param rng: generateur de rng numpy
    :return: mlp avec un hidden layer de 256 neuron
    """
    return Sequential(
        Flatten(),
        Normalize(moyenne, ecart_type),
        Linear(784, 256, rng=rng),
        ReLU(),
        # HE marche mieux pour une couche suivie d'une ReLU, lecun est mieux pour la derniere couche
        Linear(256, 10, rng=rng, init=Init.lecun_),
    )


def mlp_512_256(moyenne, ecart_type, rng):
    """
    MLP plus gros avec deux hidden layer 784 -> 512 -> 256 -> 10
    :param moyenne: moyenne du dataset
    :param ecart_type: ecart type du dataset
    :param rng: generateur de rng numpy
    :return: mlp avec deux hidden layer
    """
    return Sequential(
        Flatten(),
        Normalize(moyenne, ecart_type),
        Linear(784, 512, rng=rng),
        ReLU(),
        Linear(512, 256, rng=rng),
        ReLU(),
        Linear(256, 10, rng=rng, init=Init.lecun_),
    )

def mlp_bottleneck(moyenne, ecart_type, rng):
    return Sequential(
        Flatten(),
        Normalize(moyenne, ecart_type),
        Linear(784, 8, rng=rng),
        ReLU(),
        Linear(8, 256, rng=rng),
        ReLU(),
        Linear(256, 10, rng=rng, init=Init.lecun_),
    )

def mlp_huge(moyenne, ecart_type, rng):
    return Sequential(
        Flatten(),
        Normalize(moyenne, ecart_type),
        Linear(784, 1024, rng=rng),
        ReLU(),
        Linear(1024, 10, rng=rng, init=Init.lecun_),
    )

# ---------------------------------------------------------------------------
# Entrainement et evaluation
# ---------------------------------------------------------------------------

def train_une_epoch(model: BasicModule,
                    loader: BasicDataLoader,
                    optimizer,
                    loss_fn) -> float:
    """
    Entraine un modèle sur l'entièreté du dataset, une fois
    :param model: model a entrainer
    :param loader: dataloader pour les batchs
    :param optimizer: optimizer a utiliser
    :param loss_fn: fonction de loss a utiliser
    :return: la losse moyenne de l'epoch
    """
    model.train()
    somme, n = 0.0, 0
    for x, y in loader:
        loss = loss_fn(model(BasicTensor(x)), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Ponderation par la taille du batch car le dernier batch est plus petit
        somme += float(loss.data) * len(x)
        n += len(x)
    return somme / n


def evaluer(model: BasicModule, loader: BasicDataLoader, loss_fn) -> tuple[float, float]:
    """
    Evalue le model sur le jeu de données de validation
    :param model: modele a evaluer
    :param loader: dataloader pour les batchs de validation
    :param loss_fn: fonction de loss a utiliser
    :return: loss moyenne + accuracy du modèle sur le jeu de validation
    """
    model.eval()
    somme, justes, n = 0.0, 0, 0
    with no_grad():
        for x, y in loader:
            logits = model(BasicTensor(x))
            somme += float(loss_fn(logits, y).data) * len(x)
            justes += int((logits.data.argmax(axis=-1) == y).sum())
            n += len(x)
    model.train()
    return somme / n, justes / n


def entrainer_modele(donnees: dict,
                     modele_factory: Callable[[float, float, np.random.Generator], BasicModule] = mlp_256,
                     optim_factory: Callable [[Iterable[BasicParameter],
                                               float, ], Optimizer] = lambda p, t: AdamW(p, learning_rate=1e-3),
                     epochs: int = EPOCHS,
                     batch_size: int = BATCH_SIZE,
                     seed: int = DEFAULT_SEED,
                     loss_fn=BasicTensor.cross_entropy,
                     verbose: bool = True):
    """
    Entraine un modele avec un optimiseur pour une seed specifique.

    :param donnees: dict
        Contient les données d'entrainement et de validation
    :param modele_factory: (moyenne, ecart_type, rng) -> modele
        Instancie le modele
    :param optim_factory: (params, pas_total) -> optimiseur
        pas_total sert aux schedulers (AdEMAMix et plus tard le scheduler de lr)
    :param epochs: int
        Nombre d'epochs d'entrainement
    :param batch_size: int
        Taille des batchs
    :param seed: int
        Seed a utiliser
    :param loss_fn:
        fonction de loss a utiliser
    :param verbose: bool
        Affiche les infos
    :return (modele, historique):
        l'historique etant une liste de dicts, un par epoque
    """

    rng = np.random.default_rng(seed)
    model = modele_factory(donnees["moyenne"], donnees["ecart_type"], rng)

    x_train, y_train = donnees["train"]
    train_loader = BasicDataLoader(x_train, y_train, batch_size=batch_size, rng=rng)

    # shuffle inutile pour valider et on peut utilisr de plus gros batchs
    val_loader = BasicDataLoader(*donnees["val"], batch_size=1024, shuffle=False)

    total_step = epochs * math.ceil(len(x_train) / batch_size)
    optimizer = optim_factory(model.parameters(), total_step)

    if verbose:
        loss_init, _ = evaluer(model, val_loader, loss_fn)
        print(f"  loss initiale : {loss_init:.4f}  (log(10) = {math.log(10):.4f})")

    historique = []

    for epoch in range(epochs):
        loss_train = train_une_epoch(model, train_loader, optimizer, loss_fn)

        loss_val, acc_val = evaluer(model, val_loader, loss_fn)

        historique.append({"epoch": epoch + 1, "loss_train": loss_train,
                           "loss_val": loss_val, "acc_val": acc_val})
        if verbose:
            print(f"  epoque {epoch + 1:>2}/{epochs} | loss train {loss_train:.4f} "
                  f"| loss val {loss_val:.4f} | acc val {100 * acc_val:.2f} %")

    return model, historique


def meilleure_epoch(historique: list[dict]) -> dict:
    """Epoch avec la plus petite loss de validation"""
    return min(historique, key=lambda h: h["loss_val"])



# -----------------------------------------------------------------------
# Combinaisons de modeles et d'optim à tester
# -----------------------------------------------------------------------

MODELES = {
    "lineaire": lineaire,
    "mlp 256": mlp_256,
    #"mlp 512-256": mlp_512_256,
    #"mlp botteneck" : mlp_bottleneck,
    #"mlp huge" : mlp_huge,
}

# (params, pas_total) -> optimiseur
OPTIMISEURS = {
    #"SGD lr 0.1": lambda p, t: SGD(p, learning_rate=0.1),
    "Momentum 0.9 lr 0.01": lambda p, t: MomentumGradient(p, learning_rate=0.01, momentum=0.9),
    "AdamW lr 1e-3": lambda p, t: AdamW(p, learning_rate=1e-3),
    # lr reduit pour compenser alpha, et schedulers sur l'entrainement
    "AdEMAMix lr 2e-4": lambda p, t: AdEMAMix(p,learning_rate=2e-4,alpha=8.0,t_alpha=t, t_beta=t),
}


# ---------------------------------------------------------------------------
# Modele pre-entraine pour la demo
# ---------------------------------------------------------------------------

def get_trained_mnist(epochs: int = 5, seed: int = DEFAULT_SEED, verbose: bool = True) -> BasicModule:
    """
    Entraine et renvoie le MLP 256 avec la meilleure configuration trouvee
    (AdamW, lr 1e-3, 7 epoques), en mode eval. Reentraine a chaque appel.
    """
    donnees = preparer_donnees(val_fraction=VALIDATION_PROP, data_seed=SEED_SPLIT)
    model, _ = entrainer_modele(donnees, mlp_256, lambda p, t: AdamW(p, learning_rate=1e-3),
                                epochs=epochs, seed=seed, verbose=verbose)
    model.eval()
    return model

# -----------------------------------------------------------------------
# Test des differents modèles
# -----------------------------------------------------------------------


if __name__ == "__main__":
    donnees = preparer_donnees(val_fraction=0.1, data_seed=SEED_SPLIT)

    resultats = []

    for nom_modele, fabrique_modele in MODELES.items():
        for nom_optim, fabrique_optim in OPTIMISEURS.items():
            meilleures = []
            for seed in range(NB_SEEDS):

                print(f"\n== {nom_modele} | {nom_optim} | seed {seed}")

                _, historique = entrainer_modele(donnees, fabrique_modele, fabrique_optim, seed=seed)
                meilleures.append(meilleure_epoch(historique))

            acc = np.array([m["acc_val"] for m in meilleures])
            loss = np.array([m["loss_val"] for m in meilleures])

            epochs = [m["epoch"] for m in meilleures]

            resultats.append((nom_modele, nom_optim, acc, loss, epochs))

    # affichage de la meilleure epoch pour chaque seed pour chaque combinaison de modele et optim, + moyenne et ecart-type
    print(f"\n\nMeilleure epoch en validation sur {NB_SEEDS} graines :")
    print(f"  {'modele':<12} | {'optimiseur':<22} | {'acc val':<15} | {'loss val':<17} | epoch")

    for nom_modele, nom_optim, acc, loss, epochs in resultats:
        print(f"  {nom_modele:<12} | {nom_optim:<22} | {100 * acc.mean():5.2f} +/- {100 * acc.std():4.2f} % "
              f"| {loss.mean():.4f} +/- {loss.std():.4f} | {epochs}")


    # test d'un modèle en particulier
    model, _ = entrainer_modele(donnees, mlp_256, OPTIMISEURS["AdamW lr 1e-3"], epochs=7, seed=0)
    test_loader = BasicDataLoader(*donnees["test"], batch_size=1024, shuffle=False)
    loss_test, acc_test = evaluer(model, test_loader, BasicTensor.cross_entropy)
    print(f"TEST : acc {100 * acc_test:.2f} %, loss {loss_test:.4f}")
