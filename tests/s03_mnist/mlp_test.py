"""
Flatten et Normalize vs PyTorch

+ tests du decoupage train / validation et du chargement par minibatchs.
"""
import math

import numpy as np
import pytest

from src.s00_autograd_and_tensors.autograd import BasicTensor
from src.s01_module_and_init.layers import Flatten, Normalize, Linear, Sequential, ReLU
from src.s03_mnist.utils import split, minibatch, BasicDataLoader


def entree(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Sans PyTorch
# ---------------------------------------------------------------------------

def test_normalize_centre_et_reduit():
    x = np.random.default_rng(0).integers(0, 256, (64, 28, 28)).astype(np.float32)
    moyenne, ecart_type = x.mean(), x.std()
    sortie = Normalize(moyenne, ecart_type)(BasicTensor(x)).data
    assert abs(sortie.mean()) < 1e-4
    assert np.isclose(sortie.std(), 1.0, atol=1e-4)


def test_normalize_garde_le_float32_avec_des_statistiques_numpy():
    """
    np.mean renvoie un np.float64 => deux fois plus lent.
    """
    x = np.random.default_rng(0).integers(0, 256, (8, 28, 28)).astype(np.uint8)
    moyenne, ecart_type = np.mean(x), np.std(x)  # des np.float64
    assert isinstance(moyenne, np.float64)

    sortie = Normalize(moyenne, ecart_type)(BasicTensor(x))
    assert sortie.dtype == np.float32


def test_flatten_garde_l_axe_du_batch():
    assert Flatten()(BasicTensor(entree((5, 28, 28)))).shape == (5, 784)
    assert Flatten()(BasicTensor(entree((5, 3, 4, 4)))).shape == (5, 48)
    assert Flatten()(BasicTensor(entree((1, 28, 28)))).shape == (1, 784)


def test_flatten_parcourt_les_pixels_ligne_par_ligne():
    """L'ordre des pixels doit etre celui de NumPy (ligne par ligne), comme dans PyTorch."""
    x = entree((2, 3, 4))
    assert np.array_equal(Flatten()(BasicTensor(x)).data, x.reshape(2, -1))


# ---------------------------------------------------------------------------
# Compare a PyTorch : forward et gradient
# ---------------------------------------------------------------------------

@pytest.fixture
def torch():
    """Saute seulement les tests qui en ont besoin"""
    return pytest.importorskip("torch")


def comparer(torch, notre_module, leur_module, shape):
    """Meme sortie, meme gradient par rapport a l'entree."""
    rng = np.random.default_rng(1)
    x = entree(shape, seed=2)

    bx = BasicTensor(x, requires_grad=True)
    tx = torch.tensor(x, requires_grad=True)
    b_out = notre_module(bx)
    t_out = leur_module(tx)

    assert b_out.shape == tuple(t_out.shape)
    assert np.allclose(b_out.data, t_out.detach().numpy(), atol=1e-5)

    # Ponderation aleatoire : sinon tous les gradients amont valent 1
    w = rng.standard_normal(tuple(t_out.shape)).astype(np.float32)
    (b_out * BasicTensor(w)).sum().backward()
    (t_out * torch.tensor(w)).sum().backward()
    assert np.allclose(bx.grad, tx.grad.numpy(), atol=1e-5)


@pytest.mark.parametrize("shape", [(5, 28, 28), (5, 3, 4, 4), (1, 28, 28)])
def test_flatten_comme_nn_flatten(torch, shape):
    comparer(torch, Flatten(), torch.nn.Flatten(), shape)


def test_normalize_comme_pytorch(torch):
    """En PyTorch, la normalisation de torchvision fait exactement (x - moyenne) / ecart_type."""
    moyenne, ecart_type = 0.1307, 0.3081
    comparer(torch, Normalize(moyenne, ecart_type), lambda t: (t - moyenne) / ecart_type, (5, 28, 28))


def test_pipeline_mnist_comme_pytorch(torch):
    """Flatten + Normalize + Linear + ReLU + Linear -> le modele de l'etape 03."""
    moyenne, ecart_type = 33.3, 78.6
    torch.manual_seed(0)
    eux_1, eux_2 = torch.nn.Linear(784, 16), torch.nn.Linear(16, 10)
    nous_1, nous_2 = Linear(784, 16), Linear(16, 10)
    for nous, eux in ((nous_1, eux_1), (nous_2, eux_2)):
        nous.weight.data = eux.weight.detach().numpy().copy()
        nous.bias.data = eux.bias.detach().numpy().copy()

    nous = Sequential(Flatten(), Normalize(moyenne, ecart_type), nous_1, ReLU(), nous_2)

    class Normalisation(torch.nn.Module):
        def forward(self, t):
            return (t - moyenne) / ecart_type

    eux = torch.nn.Sequential(torch.nn.Flatten(), Normalisation(), eux_1, torch.nn.ReLU(), eux_2)

    x = np.random.default_rng(3).integers(0, 256, (8, 28, 28)).astype(np.float32)
    cibles = np.random.default_rng(4).integers(0, 10, 8)

    loss_nous = nous(BasicTensor(x)).cross_entropy(cibles)
    loss_eux = torch.nn.functional.cross_entropy(eux(torch.tensor(x)), torch.tensor(cibles, dtype=torch.long))
    loss_nous.backward()
    loss_eux.backward()

    assert np.isclose(float(loss_nous.data), loss_eux.item(), rtol=1e-4)
    for p_nous, p_eux in zip(nous.parameters(), eux.parameters()):
        assert np.allclose(p_nous.grad, p_eux.grad.numpy(), atol=1e-5)


# ---------------------------------------------------------------------------
# test etape 3
# ---------------------------------------------------------------------------


N = 103  # volontairement pas un multiple des tailles de batch testees


def donnees(n=N):
    x = np.arange(n)
    return x, 10 * x


def images(n=N):
    """images (n, 28, 28) en uint8 +  1 label/image."""
    x = np.arange(n, dtype=np.uint8).reshape(n, 1, 1) * np.ones((1, 28, 28), dtype=np.uint8)
    y = np.arange(n) % 10
    return x, y


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

def test_split_partitionne_sans_perte_ni_doublon():
    x, y = donnees()
    x_train, y_train, x_val, y_val = split(x, y, percent=0.2, rng=np.random.default_rng(0))

    assert len(set(x_train) & set(x_val)) == 0, "un exemple est a la fois en train et en validation"
    assert set(x_train) | set(x_val) == set(x), "des exemples ont disparu"
    assert len(x_train) + len(x_val) == N, "des exemples sont dupliques"


def test_split_taille_de_la_validation():
    x, y = donnees()
    _, _, x_val, _ = split(x, y, percent=0.2, rng=np.random.default_rng(0))
    assert len(x_val) == int(N * 0.2)


def test_split_garde_chaque_etiquette_avec_son_exemple():
    x, y = donnees()
    x_train, y_train, x_val, y_val = split(x, y, percent=0.2, rng=np.random.default_rng(0))
    assert np.array_equal(y_train, 10 * x_train)
    assert np.array_equal(y_val, 10 * x_val)


def test_split_melange():
    """La validation ne doit pas etre simplement les premiers exemples."""
    x, y = donnees()
    _, _, x_val, _ = split(x, y, percent=0.2, rng=np.random.default_rng(0))
    assert not np.array_equal(np.sort(x_val), np.arange(len(x_val)))


def test_split_reproductible_avec_la_meme_graine():
    x, y = donnees()
    a = split(x, y, percent=0.2, rng=np.random.default_rng(42))
    b = split(x, y, percent=0.2, rng=np.random.default_rng(42))
    for u, v in zip(a, b):
        assert np.array_equal(u, v)


def test_split_change_avec_la_graine():
    x, y = donnees()
    _, _, val_a, _ = split(x, y, percent=0.2, rng=np.random.default_rng(0))
    _, _, val_b, _ = split(x, y, percent=0.2, rng=np.random.default_rng(1))
    assert set(val_a) != set(val_b)


def test_split_sans_melange_prend_les_premiers_en_validation():
    x, y = donnees()
    x_train, _, x_val, _ = split(x, y, percent=0.2, shuffle=False)
    assert np.array_equal(x_val, np.arange(int(N * 0.2)))
    assert np.array_equal(x_train, np.arange(int(N * 0.2), N))


def test_split_garde_la_shape_des_images():
    x, y = images()
    x_train, y_train, x_val, y_val = split(x, y, percent=0.1, rng=np.random.default_rng(0))
    assert x_train.shape[1:] == (28, 28) and x_val.shape[1:] == (28, 28)
    assert x_train.dtype == np.uint8
    # chaque image vaut partout son identifiant : l'etiquette doit suivre
    assert np.array_equal(x_train[:, 0, 0] % 10, y_train)


@pytest.mark.parametrize("percent", [0, 1, -0.1, 1.5])
def test_split_refuse_une_proportion_invalide(percent):
    x, y = donnees()
    with pytest.raises(ValueError):
        split(x, y, percent=percent)


def test_split_refuse_des_longueurs_differentes():
    x, y = donnees()
    with pytest.raises(ValueError):
        split(x, y[:-1])


# ---------------------------------------------------------------------------
# minibatch
# ---------------------------------------------------------------------------

def une_epoque(lot):
    """Concatene les batchs d'une epoque."""
    batchs = list(lot)
    xs = np.concatenate([b[0] for b in batchs])
    ys = np.concatenate([b[1] for b in batchs])
    return batchs, xs, ys


@pytest.mark.parametrize("taille", [1, 10, 32, N, 1000])
def test_minibatch_voit_chaque_exemple_exactement_une_fois(taille):
    x, y = donnees()
    _, xs, _ = une_epoque(minibatch(x, y, taille, rng=np.random.default_rng(0)))
    assert np.array_equal(np.sort(xs), x)


@pytest.mark.parametrize("taille", [1, 10, 32, N, 1000])
def test_minibatch_tailles_des_batchs(taille):
    """Tous les batchs sont pleins, sauf le dernier qui prend le reste."""
    x, y = donnees()
    batchs, _, _ = une_epoque(minibatch(x, y, taille, rng=np.random.default_rng(0)))

    tailles = [len(b[0]) for b in batchs]
    assert len(tailles) == math.ceil(N / taille)
    assert all(t == taille for t in tailles[:-1])
    assert tailles[-1] == N - taille * (len(tailles) - 1)


def test_minibatch_garde_chaque_etiquette_avec_son_exemple():
    x, y = donnees()
    for xb, yb in minibatch(x, y, 16, rng=np.random.default_rng(0)):
        assert np.array_equal(yb, 10 * xb)


def test_minibatch_melange():
    x, y = donnees()
    _, xs, _ = une_epoque(minibatch(x, y, 16, rng=np.random.default_rng(0)))
    assert not np.array_equal(xs, x)


def test_minibatch_sans_melange_garde_l_ordre():
    x, y = donnees()
    _, xs, ys = une_epoque(minibatch(x, y, 16, shuffle=False))
    assert np.array_equal(xs, x)
    assert np.array_equal(ys, y)


def test_minibatch_refuse_une_taille_invalide():
    x, y = donnees()
    with pytest.raises(ValueError):
        list(minibatch(x, y, 0))  # list() : un generateur ne s'execute qu'a l'iteration


def test_minibatch_refuse_des_longueurs_differentes():
    x, y = donnees()
    with pytest.raises(ValueError):
        list(minibatch(x, y[:-1], 16))


# ---------------------------------------------------------------------------
# BasicDataLoader
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("taille", [1, 10, 32, N, 1000])
def test_dataloader_len_est_le_nombre_de_batchs(taille):
    x, y = donnees()
    loader = BasicDataLoader(x, y, batch_size=taille, rng=np.random.default_rng(0))
    assert len(loader) == math.ceil(N / taille)
    assert isinstance(len(loader), int)
    assert len(list(loader)) == len(loader)


def test_dataloader_nouvel_ordre_a_chaque_epoque():
    """Chaque `for` sur le loader est une nouvelle epoque : memes exemples, autre ordre."""
    x, y = donnees()
    loader = BasicDataLoader(x, y, batch_size=16, rng=np.random.default_rng(0))
    _, epoque_1, _ = une_epoque(loader)
    _, epoque_2, _ = une_epoque(loader)

    assert np.array_equal(np.sort(epoque_1), np.sort(epoque_2))
    assert not np.array_equal(epoque_1, epoque_2)


def test_dataloader_reproductible_avec_la_meme_graine():
    x, y = donnees()
    a = BasicDataLoader(x, y, batch_size=16, rng=np.random.default_rng(7))
    b = BasicDataLoader(x, y, batch_size=16, rng=np.random.default_rng(7))
    for _ in range(3):  # sur plusieurs epoques
        _, xa, _ = une_epoque(a)
        _, xb, _ = une_epoque(b)
        assert np.array_equal(xa, xb)


def test_dataloader_sans_melange_toujours_le_meme_ordre():
    """validation -> l'ordre ne change pas d'une epoque a l'autre."""
    x, y = donnees()
    loader = BasicDataLoader(x, y, batch_size=16, shuffle=False)
    _, epoque_1, _ = une_epoque(loader)
    _, epoque_2, _ = une_epoque(loader)
    assert np.array_equal(epoque_1, x)
    assert np.array_equal(epoque_2, x)


def test_dataloader_garde_chaque_etiquette_avec_son_image():
    x, y = images()
    loader = BasicDataLoader(x, y, batch_size=16, rng=np.random.default_rng(0))
    for xb, yb in loader:
        assert xb.shape[1:] == (28, 28)
        assert np.array_equal(xb[:, 0, 0] % 10, yb)


def test_dataloader_refuse_une_taille_invalide():
    x, y = donnees()
    with pytest.raises(ValueError):
        BasicDataLoader(x, y, batch_size=0)


def test_dataloader_refuse_des_longueurs_differentes():
    x, y = donnees()
    with pytest.raises(ValueError):
        BasicDataLoader(x, y[:-1], batch_size=16)
