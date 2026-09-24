"""
Tests pour BasicTensor (autograd).
"""
import pytest
import numpy as np

from src.s00_autograd_and_tensors.autograd import BasicTensor

def gradcheck(f, entrees, eps=1e-3, tol=1e-3):
    """
    Vérifie les calcul de dérivées via le taux d'accroissement

    f       : callable qui prend *entrees et renvoie un BasicTensor qui ne contient qu'un scalaire (shape = (,) )
    entrees : liste de BasicTensor (avec requires_grad=True)

    Compare le gradient analytique de chaque entree au gradient numerique.
    Leve une AssertionError si l'ecart depasse tol.
    """

    out = f(*entrees)
    out.backward()

    # pour chaque tensors
    for entree in entrees:
        num = np.zeros_like(entree.data) # resultat calcul
        grad_analytique = np.copy(entree.grad) # res du grad apres .backward

        # (f(x+eps) - f(x-eps)) / (2 * eps)
        for idx in np.ndindex(entree.data.shape):
            valeur = entree.data[idx]  # sauvegarde val courante

            # On bouge une valeur
            entree.data[idx] = valeur + eps

            # On vérifie l'impact sur la sortie de f
            sortie_plus = f(*entrees).data

            # On bouge dans l'autre sens
            entree.data[idx] = valeur - eps

            # On réevalue le changement sur f
            sortie_moins = f(*entrees).data

            entree.data[idx] = valeur  # on remet la valeur initiale

            # calcul numérique de la dérivée partielle
            num[idx] = (sortie_plus - sortie_moins) / (2 * eps)

        # check si les deux sont proches
        assert np.allclose(grad_analytique, num, atol = tol)



# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

def test_construction_depuis_liste():
    t = BasicTensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.data.shape == (2, 2)
    assert t.data.dtype == np.float32


def test_construction_depuis_scalaire():
    t = BasicTensor(5.0)
    assert t.data.shape == ()
    assert t.data.size == 1


def test_construction_depuis_ndarray():
    t = BasicTensor(np.arange(6.0).reshape(2, 3))
    assert t.data.shape == (2, 3)
    assert t.data.dtype == np.float32


def test_etat_initial():
    t = BasicTensor([1.0])
    assert t.grad is None
    assert t.parents == []
    assert t._backward is None
    assert t.requires_grad is False


def test_requires_grad_explicite():
    assert BasicTensor([1.0], requires_grad=True).requires_grad


# ---------------------------------------------------------------------------
# 2. Construction du graphe
# ---------------------------------------------------------------------------

def test_add_pose_les_deux_parents():
    a = BasicTensor([1.0], requires_grad=True)
    b = BasicTensor([2.0], requires_grad=True)
    c = a + b
    assert len(c.parents) == 2
    assert c.parents[0] is a
    assert c.parents[1] is b


def test_mul_pose_les_deux_parents():
    a = BasicTensor([1.0], requires_grad=True)
    b = BasicTensor([2.0], requires_grad=True)
    c = a * b
    assert c.parents[0] is a and c.parents[1] is b


def test_requires_grad_se_propage():
    a = BasicTensor([1.0], requires_grad=True)
    b = BasicTensor([2.0])
    assert (a + b).requires_grad
    assert (b + a).requires_grad


def test_pas_de_graphe_sans_requires_grad():
    a = BasicTensor([1.0])
    b = BasicTensor([2.0])
    c = a + b
    assert not c.requires_grad
    assert c.parents == []
    assert c._backward is None


def test_scalaire_python_est_enveloppe():
    a = BasicTensor([1.0, 2.0], requires_grad=True)
    c = a + 3.0
    assert isinstance(c.parents[1], BasicTensor)
    assert np.allclose(c.data, [4.0, 5.0])


def test_valeurs_forward():
    a = BasicTensor([[1.0, 2.0], [3.0, 4.0]])
    b = BasicTensor([[5.0, 6.0], [7.0, 8.0]])
    assert np.allclose((a + b).data, [[6.0, 8.0], [10.0, 12.0]])
    assert np.allclose((a * b).data, [[5.0, 12.0], [21.0, 32.0]])


def test_forward_avec_broadcast():
    a = BasicTensor(np.ones((3, 4)))
    b = BasicTensor([1.0, 2.0, 3.0, 4.0])
    assert (a + b).data.shape == (3, 4)


# ---------------------------------------------------------------------------
# 3. Tri topologique
# ---------------------------------------------------------------------------

def test_ordre_commence_par_la_racine():
    a = BasicTensor([1.0], requires_grad=True)
    b = BasicTensor([2.0], requires_grad=True)
    c = a + b
    ordre = c._backward_order()
    assert ordre[0] is c


def test_ordre_contient_tout_le_graphe():
    a = BasicTensor([1.0], requires_grad=True)
    b = BasicTensor([2.0], requires_grad=True)
    c = a * b
    d = c + a
    ordre = d._backward_order()
    assert set(id(t) for t in ordre) == {id(a), id(b), id(c), id(d)}


def test_ordre_sans_doublon():
    a = BasicTensor([1.0], requires_grad=True)
    b = BasicTensor([2.0], requires_grad=True)
    c = BasicTensor([3.0], requires_grad=True)
    d = (a * b) + (a * c)
    ordre = d._backward_order()
    assert len(ordre) == len(set(id(t) for t in ordre))


def test_ordre_parent_apres_enfant():
    a = BasicTensor([1.0], requires_grad=True)
    b = BasicTensor([2.0], requires_grad=True)
    c = a * b
    d = c + a
    ordre = d._backward_order()
    pos = {id(t): i for i, t in enumerate(ordre)}
    assert pos[id(d)] < pos[id(c)]
    assert pos[id(c)] < pos[id(a)]
    assert pos[id(c)] < pos[id(b)]


def test_ordre_ignore_sans_requires_grad():
    a = BasicTensor([1.0], requires_grad=True)
    b = BasicTensor([2.0])          # pas de requires_grad
    c = a + b
    ordre = c._backward_order()
    assert all(t is not b for t in ordre)


# ---------------------------------------------------------------------------
# 4. Backward : cas simples
# ---------------------------------------------------------------------------

def test_backward_refuse_non_scalaire():
    a = BasicTensor([1.0, 2.0], requires_grad=True)
    b = BasicTensor([3.0, 4.0], requires_grad=True)
    with pytest.raises(ValueError):
        (a + b).backward()


def test_backward_grad_racine_vaut_un():
    a = BasicTensor(1.0, requires_grad=True)
    b = BasicTensor(2.0, requires_grad=True)
    c = a + b
    c.backward()
    assert np.allclose(c.grad, 1.0)


def test_backward_add():
    """d(a+b)/da = 1, d(a+b)/db = 1"""
    a = BasicTensor(3.0, requires_grad=True)
    b = BasicTensor(4.0, requires_grad=True)
    (a + b).backward()
    assert np.allclose(a.grad, 1.0)
    assert np.allclose(b.grad, 1.0)


def test_backward_mul():
    """d(a*b)/da = b, d(a*b)/db = a"""
    a = BasicTensor(3.0, requires_grad=True)
    b = BasicTensor(4.0, requires_grad=True)
    (a * b).backward()
    assert np.allclose(a.grad, 4.0)
    assert np.allclose(b.grad, 3.0)


def test_backward_regle_de_la_chaine():
    """d((a*b)*c)/da = b*c"""
    a = BasicTensor(2.0, requires_grad=True)
    b = BasicTensor(3.0, requires_grad=True)
    c = BasicTensor(4.0, requires_grad=True)
    ((a * b) * c).backward()
    assert np.allclose(a.grad, 12.0)
    assert np.allclose(b.grad, 8.0)
    assert np.allclose(c.grad, 6.0)


def test_backward_pas_de_grad_sans_requires_grad():
    a = BasicTensor(3.0, requires_grad=True)
    b = BasicTensor(4.0)
    (a * b).backward()
    assert a.grad is not None
    assert b.grad is None


# ---------------------------------------------------------------------------
# 5. Backward : accumulation
# ---------------------------------------------------------------------------

def test_noeud_partage_accumule():
    a = BasicTensor(3.0, requires_grad=True)
    (a * a).backward()
    assert np.allclose(a.grad, 6.0)


def test_noeud_partage_deux_branches():
    a = BasicTensor(2.0, requires_grad=True)
    b = BasicTensor(3.0, requires_grad=True)
    c = BasicTensor(5.0, requires_grad=True)
    ((a * b) + (a * c)).backward()
    assert np.allclose(a.grad, 8.0)


def test_a_plus_a():
    a = BasicTensor(3.0, requires_grad=True)
    (a + a).backward()
    assert np.allclose(a.grad, 2.0)


def test_backward_deux_fois_accumule():
    a = BasicTensor(3.0, requires_grad=True)
    b = BasicTensor(4.0, requires_grad=True)
    (a * b).backward()
    premier = a.grad.copy()
    (a * b).backward()
    assert np.allclose(a.grad, 2 * premier)


# ---------------------------------------------------------------------------
# 6. Backward avec broadcast
# ---------------------------------------------------------------------------


def test_backward_biais_broadcast():
    a = BasicTensor(np.zeros((3, 4)), requires_grad=True)
    b = BasicTensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    out = a + b
    out.grad = np.ones((3, 4), dtype=np.float32)
    out._backward()
    assert b.grad.shape == (4,)
    assert np.allclose(b.grad, [3.0, 3.0, 3.0, 3.0])
    assert a.grad.shape == (3, 4)


def test_backward_scalaire_broadcast():
    a = BasicTensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    s = BasicTensor(2.0, requires_grad=True)
    out = a * s
    out.grad = np.ones((2, 3), dtype=np.float32)
    out._backward()
    assert s.grad.shape == ()
    assert np.allclose(s.grad, np.arange(6.0).sum())


# ---------------------------------------------------------------------------
# 7. Comparaison avec PyTorch
# ---------------------------------------------------------------------------

def test_contre_pytorch_expression_simple():
    torch = pytest.importorskip("torch")

    ta = torch.tensor(2.0, requires_grad=True)
    tb = torch.tensor(3.0, requires_grad=True)
    ((ta * tb) + ta).backward()

    a = BasicTensor(2.0, requires_grad=True)
    b = BasicTensor(3.0, requires_grad=True)
    ((a * b) + a).backward()

    assert np.allclose(a.grad, ta.grad.numpy())
    assert np.allclose(b.grad, tb.grad.numpy())


def test_contre_pytorch_noeud_partage():
    torch = pytest.importorskip("torch")

    ta = torch.tensor(2.0, requires_grad=True)
    tb = torch.tensor(3.0, requires_grad=True)
    tc = torch.tensor(5.0, requires_grad=True)
    ((ta * tb) + (ta * tc)).backward()

    a = BasicTensor(2.0, requires_grad=True)
    b = BasicTensor(3.0, requires_grad=True)
    c = BasicTensor(5.0, requires_grad=True)
    ((a * b) + (a * c)).backward()

    assert np.allclose(a.grad, ta.grad.numpy())
    assert np.allclose(b.grad, tb.grad.numpy())
    assert np.allclose(c.grad, tc.grad.numpy())


# ---------------------------------------------------------------------------
# 8. sum
# ---------------------------------------------------------------------------

def test_sum_forward():
    a = BasicTensor([[1.0, 2.0], [3.0, 4.0]])
    assert a.sum().data.shape == ()
    assert np.allclose(a.sum().data, 10.0)


def test_sum_backward_un_partout():
    a = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    a.sum().backward()
    assert a.grad.shape == (3, 4)
    assert np.allclose(a.grad, np.ones((3, 4)))


def test_sum_pose_un_seul_parent():
    a = BasicTensor([1.0, 2.0], requires_grad=True)
    s = a.sum()
    assert len(s.parents) == 1
    assert s.parents[0] is a


def test_sum_sans_requires_grad():
    a = BasicTensor([1.0, 2.0])
    s = a.sum()
    assert not s.requires_grad
    assert s._backward is None


def test_sum_scalaire():
    a = BasicTensor(5.0, requires_grad=True)
    s = a.sum()
    s.backward()
    assert np.allclose(s.data, 5.0)
    assert np.allclose(a.grad, 1.0)


def test_sum_dtype_preserve():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    a.sum().backward()
    assert a.grad.dtype == np.float32


def test_sum_apres_mul():
    """d(sum(a*b))/da = b"""
    a = BasicTensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = BasicTensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    (a * b).sum().backward()
    assert np.allclose(a.grad, b.data)
    assert np.allclose(b.grad, a.data)


def test_sum_contre_pytorch():
    torch = pytest.importorskip("torch")
    donnees = np.random.randn(3, 4).astype(np.float32)

    ta = torch.tensor(donnees, requires_grad=True)
    (ta * ta).sum().backward()

    a = BasicTensor(donnees, requires_grad=True)
    (a * a).sum().backward()

    assert np.allclose(a.grad, ta.grad.numpy(), atol=1e-5)


# ---------------------------------------------------------------------------
# 9. gradcheck
# ---------------------------------------------------------------------------

def test_gradcheck_mul_scalaires():
    """Valide gradcheck lui-meme sur le cas le plus simple."""
    a = BasicTensor(2.0, requires_grad=True)
    b = BasicTensor(3.0, requires_grad=True)
    gradcheck(lambda x, y: x * y, [a, b])


def test_gradcheck_add_scalaires():
    a = BasicTensor(2.0, requires_grad=True)
    b = BasicTensor(3.0, requires_grad=True)
    gradcheck(lambda x, y: x + y, [a, b])


def test_gradcheck_add():
    a = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    b = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    gradcheck(lambda x, y: (x + y).sum(), [a, b])


def test_gradcheck_mul():
    a = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    b = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    gradcheck(lambda x, y: (x * y).sum(), [a, b])


def test_gradcheck_expression_mixte():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    b = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    gradcheck(lambda x, y: ((x * y) + x).sum(), [a, b])


def test_gradcheck_avec_broadcast():
    a = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    b = BasicTensor(np.random.randn(4), requires_grad=True)
    gradcheck(lambda x, y: (x + y).sum(), [a, b])


def test_gradcheck_broadcast_mul():
    a = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    b = BasicTensor(np.random.randn(4), requires_grad=True)
    gradcheck(lambda x, y: (x * y).sum(), [a, b])


def test_gradcheck_broadcast_scalaire():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    s = BasicTensor(2.0, requires_grad=True)
    gradcheck(lambda x, y: (x * y).sum(), [a, s])


def test_gradcheck_broadcast_keepdims():
    a = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    b = BasicTensor(np.random.randn(3, 1), requires_grad=True)
    gradcheck(lambda x, y: (x + y).sum(), [a, b])


def test_gradcheck_noeud_partage():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    b = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    c = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    gradcheck(lambda x, y, z: ((x * y) + (x * z)).sum(), [a, b, c])


def test_gradcheck_carre():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    gradcheck(lambda x: (x * x).sum(), [a])


def test_gradcheck_chaine_profonde():
    a = BasicTensor(np.random.randn(2, 2), requires_grad=True)
    b = BasicTensor(np.random.randn(2, 2), requires_grad=True)
    gradcheck(lambda x, y: (((x * y) + x) * (x + y)).sum(), [a, b])


def test_gradcheck_avec_constante():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    gradcheck(lambda x: ((x * 3.0) + 2.0).sum(), [a])


def test_gradcheck_une_seule_entree_derivable():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    b = BasicTensor(np.random.randn(2, 3))
    gradcheck(lambda x: (x * b).sum(), [a])
    assert b.grad is None
