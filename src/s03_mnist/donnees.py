"""
Chargement de MNIST sans importer tensorflow ou torchvision
"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

URL_MNIST = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

# stock dans data/
DOSSIER_DONNEES = Path(__file__).resolve().parents[2] / "data"


def charger_mnist(dossier: Path = DOSSIER_DONNEES):
    """
    Renvoie MNIST sans nettoyage ou traitement
    Cette fonction a été générée à l'aide de l'IA générative

    :param dossier: où garder le fichier en cache
    :return: (x_train, y_train, x_test, y_test)
        x_train (60000, 28, 28), y_train (60000,), x_test (10000, 28, 28), y_test (10000,)
    """
    dossier = Path(dossier)
    chemin = dossier / "mnist.npz"

    if not chemin.exists():
        dossier.mkdir(parents=True, exist_ok=True)
        print(f"Téléchargement de MNIST dans {chemin} ...")
        # telechargement dans fichier temporaire au cas ou le telechargement echoue
        temporaire = chemin.with_suffix(".tmp")
        urlretrieve(URL_MNIST, temporaire)
        temporaire.replace(chemin)

    with np.load(chemin) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    assert x_train.shape == (60000, 28, 28) and y_train.shape == (60000,)
    assert x_test.shape == (10000, 28, 28) and y_test.shape == (10000,)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = charger_mnist()
    print(f"train : {x_train.shape} {x_train.dtype}, test : {x_test.shape}")
    print(f"pixels entre {x_train.min()} et {x_train.max()}")
    print(f"exemples par classe (train) : {np.bincount(y_train)}")
