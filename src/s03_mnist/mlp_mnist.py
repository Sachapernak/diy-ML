from typing import Callable

import numpy as np


from src.s00_autograd_and_tensors import autograd
from src.s01_module_and_init.module import BasicModule
from src.s03_mnist.utils import BasicDataLoader
from src.s01_module_and_init.init import Init
from src.s02_optimizer.optimizer import AdamW, AdEMAMix, SGD
from src.s00_autograd_and_tensors.autograd import BasicTensor
from src.s03_mnist.utils import split
from src.s01_module_and_init.layers import Sequential, Flatten, Normalize, Linear, ReLU
from src.s03_mnist.donnees import charger_mnist



# Hyperparametres
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
epochs = 7

def training_loop(dataloader : BasicDataLoader, model : BasicModule, loss_fn, optimizer, progress_frequency = 100 ):
    """
    Boucle d'entrainement qui reprent celle de pytorch

    :param dataloader: dataloader qui iter des batchs
    :param model: model a entrainer
    :param loss_fn: fonction de loss (lambda)
    :param optimizer: optimiseur a utiliser (lambda)
    :param progress_frequency: frequence d'affichage de la loss
    :return:
    """
    size = dataloader.dataset_size()
    batch_size = dataloader.batch_size

    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x = BasicTensor(x)
        predictions = model(x)
        loss = loss_fn(predictions, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % progress_frequency == 0:
            loss = loss.data
            current = batch * dataloader.batch_size + len(x)
            print(f"loss : {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader : BasicDataLoader, model : BasicModule, loss_fn):
    model.eval()
    size = dataloader.dataset_size()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with autograd.no_grad():
        for x, y in dataloader:
            x = BasicTensor(x)
            pred = model(x)
            test_loss += loss_fn(pred, y).item() * len(x)
            correct += (pred.argmax(1) == y).sum().item()

    test_loss /= size
    correct /= size
    print(f"Resultat du test : "
          f"exactitude (accuracy) : {(100*correct):>0.1f}%, loss moyenne : {test_loss:>7f} \n")


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = charger_mnist()

    # seed pour la reproductibilité
    rng = np.random.default_rng(1337)

    # découpage pour la validation, avec 10% pour valider
    x_train, y_train, x_val, y_val = split(x_train, y_train, percent=0.1, rng=rng)

    # calcul de la moyenne et de la variance pour la normalisation
    moyenne_mnist = np.mean(x_train)
    std_mnist = np.std(x_train)


    # calcul dee la taille des entrées après flatten (784)
    in_size = x_train[0].size

    mnist_mlp = Sequential(
        Flatten(),
        Normalize(moyenne_mnist, std_mnist),
        Linear(in_features=in_size, out_features= 256, rng=rng),
        ReLU(),
        # HE est plutot fait pour une couche avec un ReLU après du coup vaut mieux autre chose (ici lecun qui
        # a une variance plus petite, 1 vs 2)
        Linear(in_features= 256, out_features= 10, rng=rng, init=Init.lecun_),
    )

    loss_fn = BasicTensor.cross_entropy
    optimizer = AdamW(mnist_mlp.parameters(), learning_rate=LEARNING_RATE)


    # calcul de la losse initiale
    loss_intiale = mnist_mlp(BasicTensor(x_train[:64])).cross_entropy(y_train[:64])
    print(f"Loss initiale avant entrainement : {float(loss_intiale.data)}")

    training_dataloader = BasicDataLoader(x_train, y_train, batch_size=BATCH_SIZE, rng=rng)
    validation_dataloader = BasicDataLoader(x_val, y_val, batch_size=BATCH_SIZE, rng=rng)

    for epoch in range(epochs):
        print("== Epoch: ", epoch)
        training_loop(training_dataloader, mnist_mlp, loss_fn , optimizer)
        test_loop(validation_dataloader, mnist_mlp, loss_fn)





