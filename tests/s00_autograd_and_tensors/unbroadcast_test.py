"""
Tests pour unbroadcast.
"""

import pytest
import numpy as np

from src.s00_autograd_and_tensors.autograd import unbroadcast


# ---------------------------------------------------------------------------
# 1. Shapes
# ---------------------------------------------------------------------------

def test_axes_en_trop():
    assert unbroadcast(np.ones((32, 10)), (10,)).shape == (10,)


def test_dimension_a_un_avec_keepdims():
    assert unbroadcast(np.ones((32, 10)), (1, 10)).shape == (1, 10)


def test_les_deux_mecanismes():
    # (5,4,3) -> (1,3) : un axe en trop a gauche, puis une dim a 1
    assert unbroadcast(np.ones((5, 4, 3)), (1, 3)).shape == (1, 3)


def test_vers_scalaire():
    assert unbroadcast(np.ones((32, 10)), ()).shape == ()


def test_shape_identique_ne_change_rien():
    grad = np.arange(6.0).reshape(2, 3)
    res = unbroadcast(grad, (2, 3))
    assert res.shape == (2, 3)
    assert np.allclose(res, grad)


def test_plusieurs_axes_en_trop():
    assert unbroadcast(np.ones((2, 3, 4, 5)), (5,)).shape == (5,)


def test_dim_a_un_sur_dernier_axe():
    assert unbroadcast(np.ones((4, 6)), (4, 1)).shape == (4, 1)


def test_toutes_dims_a_un():
    assert unbroadcast(np.ones((3, 4)), (1, 1)).shape == (1, 1)


# ---------------------------------------------------------------------------
# 2. Valeurs
# ---------------------------------------------------------------------------

def test_somme_des_repetitions():
    """Chaque element a ete repete 32 fois -> recoit 32 contributions."""
    res = unbroadcast(np.ones((32, 10)), (10,))
    assert np.allclose(res, np.full((10,), 32.0))


def test_somme_vers_scalaire():
    res = unbroadcast(np.ones((4, 5)), ())
    assert np.allclose(res, 20.0)


def test_somme_keepdims():
    res = unbroadcast(np.ones((7, 3)), (1, 3))
    assert np.allclose(res, np.full((1, 3), 7.0))


def test_somme_axe_interne():
    res = unbroadcast(np.ones((4, 6)), (4, 1))
    assert np.allclose(res, np.full((4, 1), 6.0))


def test_valeurs_non_uniformes():
    """[[0,1,2],[3,4,5]] somme sur l'axe 0 -> [3, 5, 7]"""
    grad = np.arange(6.0).reshape(2, 3)
    res = unbroadcast(grad, (3,))
    assert np.allclose(res, np.array([3.0, 5.0, 7.0]))


def test_valeurs_axe_interne_non_uniformes():
    """[[0,1,2],[3,4,5]] somme sur l'axe 1 avec keepdims -> [[3],[12]]"""
    grad = np.arange(6.0).reshape(2, 3)
    res = unbroadcast(grad, (2, 1))
    assert np.allclose(res, np.array([[3.0], [12.0]]))


def test_somme_totale_conservee():
    """Invariant fort : unbroadcast redistribue, il ne perd rien."""
    grad = np.random.randn(4, 5, 3)
    for cible in [(3,), (5, 3), (1, 3), (1, 1, 3), ()]:
        res = unbroadcast(grad, cible)
        assert np.isclose(res.sum(), grad.sum())


# ---------------------------------------------------------------------------
# 3. Coherence (broadcast puis unbroadcast)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("petite,grande", [
    ((10,), (32, 10)),
    ((1, 10), (32, 10)),
    ((3,), (5, 4, 3)),
    ((1, 3), (4, 3)),
    ((4, 1), (4, 6)),
    ((), (2, 3)),
    ((2, 3), (2, 3)),
])
def test_aller_retour_shape(petite, grande):
    """Ce que np.broadcast_to etend, unbroadcast doit le ramener."""
    etendu = np.broadcast_to(np.ones(petite), grande)
    assert unbroadcast(np.asarray(etendu), petite).shape == petite


@pytest.mark.parametrize("petite,grande", [
    ((10,), (32, 10)),
    ((1, 10), (32, 10)),
    ((3,), (5, 4, 3)),
    ((4, 1), (4, 6)),
])
def test_facteur_de_repetition(petite, grande):
    """Le resultat vaut le nombre de repetitions, partout."""
    facteur = int(np.prod(grande)) // int(np.prod(petite))
    res = unbroadcast(np.ones(grande), petite)
    assert np.allclose(res, np.full(petite, float(facteur)))


# ---------------------------------------------------------------------------
# 4. Comparaison avec PyTorch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape_a,shape_b", [
    ((32, 10), (10,)),
    ((32, 10), (1, 10)),
    ((5, 4, 3), (3,)),
    ((4, 6), (4, 1)),
    ((2, 3), ()),
])
def test_contre_pytorch(shape_a, shape_b):
    torch = pytest.importorskip("torch")

    a = torch.zeros(shape_a, requires_grad=True)
    b = torch.zeros(shape_b, requires_grad=True)
    (a + b).sum().backward()

    attendu = b.grad.numpy()
    obtenu = unbroadcast(np.ones(shape_a), shape_b)
    assert obtenu.shape == attendu.shape
    assert np.allclose(obtenu, attendu)


# ---------------------------------------------------------------------------
# 5. Erreurs
# ---------------------------------------------------------------------------

def test_rejette_shape_plus_grande():
    with pytest.raises(ValueError):
        unbroadcast(np.ones((10,)), (32, 10))


def test_rejette_dimension_incompatible():
    with pytest.raises(ValueError):
        unbroadcast(np.ones((32, 10)), (32, 7))


def test_rejette_dimension_incompatible_dernier_axe():
    with pytest.raises(ValueError):
        unbroadcast(np.ones((5, 4, 3)), (4, 5))


# ---------------------------------------------------------------------------
# 6. Le cas reel : x @ W + b
# ---------------------------------------------------------------------------

def test_cas_biais_mlp():
    """b de shape (10,) additionne a une sortie (32, 10) : c'est LE cas d'usage."""
    batch, features = 32, 10
    grad_sortie = np.random.randn(batch, features)
    grad_biais = unbroadcast(grad_sortie, (features,))

    assert grad_biais.shape == (features,)
    for j in range(features):
        assert np.isclose(grad_biais[j], grad_sortie[:, j].sum())