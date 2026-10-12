"""
Tests pour BasicTensor (autograd).
"""
import pytest
import numpy as np

from src.s00_autograd_and_tensors.autograd import BasicTensor

def gradcheck(f, entrees, eps=1e-2, tol=1e-2):
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


# ---------------------------------------------------------------------------
# 10. neg / sub
# ---------------------------------------------------------------------------

def test_neg_forward():
    a = BasicTensor([[1.0, -2.0], [3.0, -4.0]])
    assert np.allclose((-a).data, [[-1.0, 2.0], [-3.0, 4.0]])


def test_neg_backward():
    a = BasicTensor(3.0, requires_grad=True)
    (-a).backward()
    assert np.allclose(a.grad, -1.0)


def test_gradcheck_neg():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    gradcheck(lambda x: (-x).sum(), [a])


def test_sub_forward():
    a = BasicTensor([5.0, 7.0])
    b = BasicTensor([2.0, 3.0])
    assert np.allclose((a - b).data, [3.0, 4.0])


def test_sub_backward_signes_opposes():
    a = BasicTensor(5.0, requires_grad=True)
    b = BasicTensor(2.0, requires_grad=True)
    (a - b).backward()
    assert np.allclose(a.grad, 1.0)
    assert np.allclose(b.grad, -1.0)


def test_gradcheck_sub():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    b = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    gradcheck(lambda x, y: (x - y).sum(), [a, b])


def test_gradcheck_sub_broadcast():
    a = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    b = BasicTensor(np.random.randn(4), requires_grad=True)
    gradcheck(lambda x, y: (x - y).sum(), [a, b])


# ---------------------------------------------------------------------------
# 11. matmul
# ---------------------------------------------------------------------------

def test_matmul_forward():
    a = BasicTensor([[1.0, 2.0], [3.0, 4.0]])
    b = BasicTensor([[5.0, 6.0], [7.0, 8.0]])
    assert np.allclose((a @ b).data, [[19.0, 22.0], [43.0, 50.0]])


def test_matmul_shape():
    a = BasicTensor(np.random.randn(2, 3))
    b = BasicTensor(np.random.randn(3, 4))
    assert (a @ b).data.shape == (2, 4)


def test_matmul_backward_shapes():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    b = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    (a @ b).sum().backward()
    assert a.grad.shape == (2, 3)
    assert b.grad.shape == (3, 4)


def test_matmul_backward_valeurs():
    A = np.random.randn(2, 3).astype(np.float32)
    B = np.random.randn(3, 4).astype(np.float32)
    a = BasicTensor(A, requires_grad=True)
    b = BasicTensor(B, requires_grad=True)
    (a @ b).sum().backward()
    ones = np.ones((2, 4), dtype=np.float32)
    assert np.allclose(a.grad, ones @ B.T, atol=1e-5)
    assert np.allclose(b.grad, A.T @ ones, atol=1e-5)


def test_gradcheck_matmul():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    b = BasicTensor(np.random.randn(3, 2), requires_grad=True)
    gradcheck(lambda x, y: (x @ y).sum(), [a, b])


def test_gradcheck_matmul_non_carre():
    a = BasicTensor(np.random.randn(3, 5), requires_grad=True)
    b = BasicTensor(np.random.randn(5, 2), requires_grad=True)
    gradcheck(lambda x, y: (x @ y).sum(), [a, b])


def test_matmul_contre_pytorch():
    torch = pytest.importorskip("torch")
    A = np.random.randn(3, 4).astype(np.float32)
    B = np.random.randn(4, 2).astype(np.float32)

    ta = torch.tensor(A, requires_grad=True)
    tb = torch.tensor(B, requires_grad=True)
    (ta @ tb).sum().backward()

    a = BasicTensor(A, requires_grad=True)
    b = BasicTensor(B, requires_grad=True)
    (a @ b).sum().backward()

    assert np.allclose(a.grad, ta.grad.numpy(), atol=1e-4)
    assert np.allclose(b.grad, tb.grad.numpy(), atol=1e-4)


def test_matmul_couche_lineaire():
    x = BasicTensor(np.random.randn(8, 4), requires_grad=True)
    W = BasicTensor(np.random.randn(4, 3), requires_grad=True)
    b = BasicTensor(np.random.randn(3), requires_grad=True)
    ((x @ W) + b).sum().backward()
    assert x.grad.shape == (8, 4)
    assert W.grad.shape == (4, 3)
    assert b.grad.shape == (3,)
    assert np.allclose(b.grad, 8.0)  # 8 lignes de batch


# ---------------------------------------------------------------------------
# 12. pow / div
# ---------------------------------------------------------------------------

def test_pow_forward():
    a = BasicTensor([2.0, 3.0])
    assert np.allclose(a.pow(2.0).data, [4.0, 9.0])


def test_pow_backward_exposant_constant():
    """d(a^2)/da = 2a"""
    a = BasicTensor(3.0, requires_grad=True)
    a.pow(2.0).backward()
    assert np.allclose(a.grad, 6.0)


def test_gradcheck_pow_carre():
    a = BasicTensor(np.random.rand(2, 3) + 1.0, requires_grad=True)  # > 0
    gradcheck(lambda x: x.pow(2.0).sum(), [a])


def test_gradcheck_pow_inverse():
    a = BasicTensor(np.random.rand(2, 3) + 1.0, requires_grad=True)
    gradcheck(lambda x: x.pow(-1.0).sum(), [a])


def test_div_forward():
    a = BasicTensor([6.0, 8.0])
    b = BasicTensor([2.0, 4.0])
    assert np.allclose((a / b).data, [3.0, 2.0])


def test_div_backward():
    a = BasicTensor(6.0, requires_grad=True)
    b = BasicTensor(2.0, requires_grad=True)
    (a / b).backward()
    assert np.allclose(a.grad, 0.5, atol=1e-5)
    assert np.allclose(b.grad, -1.5, atol=1e-5)


def test_gradcheck_div():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    b = BasicTensor(np.random.rand(2, 3) + 1.0, requires_grad=True)  # loin de 0
    gradcheck(lambda x, y: (x / y).sum(), [a, b])


# ---------------------------------------------------------------------------
# 13. relu
# ---------------------------------------------------------------------------

def test_relu_forward():
    a = BasicTensor([-2.0, -0.5, 0.0, 1.0, 3.0])
    assert np.allclose(a.relu().data, [0.0, 0.0, 0.0, 1.0, 3.0])


def test_relu_backward_est_une_indicatrice():
    a = BasicTensor([-2.0, 5.0, 3.0], requires_grad=True)
    a.relu().sum().backward()
    assert np.allclose(a.grad, [0.0, 1.0, 1.0])


def test_relu_backward_zero_exclu():
    a = BasicTensor([0.0], requires_grad=True)
    a.relu().sum().backward()
    assert np.allclose(a.grad, [0.0])


def test_gradcheck_relu():
    donnees = np.random.randn(2, 4)
    donnees = np.where(np.abs(donnees) < 0.1, 0.5, donnees)
    a = BasicTensor(donnees, requires_grad=True)
    gradcheck(lambda x: x.relu().sum(), [a])


def test_relu_contre_pytorch():
    torch = pytest.importorskip("torch")
    donnees = np.array([[-2.0, 1.0], [0.5, -0.3]], dtype=np.float32)

    ta = torch.tensor(donnees, requires_grad=True)
    torch.relu(ta).sum().backward()

    a = BasicTensor(donnees, requires_grad=True)
    a.relu().sum().backward()

    assert np.allclose(a.grad, ta.grad.numpy())


# ---------------------------------------------------------------------------
# 14. maximum (element par element)
# ---------------------------------------------------------------------------

def test_maximum_forward():
    a = BasicTensor([1.0, 5.0, 3.0])
    b = BasicTensor([4.0, 2.0, 3.0])
    assert np.allclose(a.maximum(b).data, [4.0, 5.0, 3.0])


def test_maximum_backward_va_au_plus_grand():
    a = BasicTensor([1.0, 5.0], requires_grad=True)
    b = BasicTensor([4.0, 2.0], requires_grad=True)
    a.maximum(b).sum().backward()
    assert np.allclose(a.grad, [0.0, 1.0])
    assert np.allclose(b.grad, [1.0, 0.0])


def test_maximum_egalite_un_seul_recoit():
    a = BasicTensor([3.0], requires_grad=True)
    b = BasicTensor([3.0], requires_grad=True)
    a.maximum(b).sum().backward()
    total = float(a.grad[0]) + float(b.grad[0])
    assert np.isclose(total, 1.0)


def test_gradcheck_maximum():
    a = BasicTensor(np.random.randn(2, 3) + 5.0, requires_grad=True)
    b = BasicTensor(np.random.randn(2, 3) - 5.0, requires_grad=True)
    gradcheck(lambda x, y: x.maximum(y).sum(), [a, b], eps=1e-2, tol=1e-2)


def test_maximum_broadcast():
    a = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    z = BasicTensor(0.0)
    out = a.maximum(z)
    assert out.data.shape == (3, 4)


# ---------------------------------------------------------------------------
# 15. max (reduction)
# ---------------------------------------------------------------------------

def test_max_global_forward():
    a = BasicTensor([[1.0, 7.0], [3.0, 2.0]])
    assert np.allclose(a.max().data, 7.0)
    assert a.max().data.shape == ()


def test_max_axe_forward():
    a = BasicTensor([[1.0, 7.0], [3.0, 2.0]])
    assert np.allclose(a.max(axis=1).data, [7.0, 3.0])
    assert np.allclose(a.max(axis=0).data, [3.0, 7.0])


def test_max_keepdims():
    a = BasicTensor([[1.0, 7.0], [3.0, 2.0]])
    assert a.max(axis=1, keepdims=True).data.shape == (2, 1)


def test_max_global_backward():
    a = BasicTensor([[1.0, 7.0], [3.0, 2.0]], requires_grad=True)
    a.max().backward()
    assert np.allclose(a.grad, [[0.0, 1.0], [0.0, 0.0]])


def test_max_axe_backward():
    a = BasicTensor([[1.0, 7.0], [3.0, 2.0]], requires_grad=True)
    a.max(axis=1).sum().backward()
    assert np.allclose(a.grad, [[0.0, 1.0], [1.0, 0.0]])


def test_max_keepdims_backward():
    a = BasicTensor([[1.0, 7.0], [3.0, 2.0]], requires_grad=True)
    a.max(axis=1, keepdims=True).sum().backward()
    assert np.allclose(a.grad, [[0.0, 1.0], [1.0, 0.0]])


def test_max_grad_shape_preservee():
    a = BasicTensor(np.random.randn(3, 4, 5), requires_grad=True)
    a.max(axis=1).sum().backward()
    assert a.grad.shape == (3, 4, 5)


def test_max_contre_pytorch():
    torch = pytest.importorskip("torch")
    donnees = np.random.randn(3, 4).astype(np.float32)

    ta = torch.tensor(donnees, requires_grad=True)
    ta.max(dim=1).values.sum().backward()

    a = BasicTensor(donnees, requires_grad=True)
    a.max(axis=1).sum().backward()

    assert np.allclose(a.grad, ta.grad.numpy())


def test_max_dtype():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    a.max(axis=1).sum().backward()
    assert a.grad.dtype == np.float32


# ---------------------------------------------------------------------------
# 16. Compositions
# ---------------------------------------------------------------------------

def test_gradcheck_couche_lineaire_relu():
    x = BasicTensor(np.random.randn(4, 3), requires_grad=True)
    W = BasicTensor(np.random.randn(3, 2), requires_grad=True)
    b = BasicTensor(np.random.randn(2), requires_grad=True)
    gradcheck(lambda a, w, c: ((a @ w) + c).relu().sum(), [x, W, b])


def test_gradcheck_deux_couches():
    x = BasicTensor(np.random.randn(4, 3), requires_grad=True)
    W1 = BasicTensor(np.random.randn(3, 5), requires_grad=True)
    W2 = BasicTensor(np.random.randn(5, 2), requires_grad=True)
    gradcheck(lambda a, w1, w2: ((a @ w1).relu() @ w2).sum(), [x, W1, W2], tol=1e-2, eps=1e-2)


def test_mlp_contre_pytorch():
    torch = pytest.importorskip("torch")
    X = np.random.randn(4, 3).astype(np.float32)
    W1 = np.random.randn(3, 5).astype(np.float32)
    W2 = np.random.randn(5, 2).astype(np.float32)

    tx = torch.tensor(X, requires_grad=True)
    tw1 = torch.tensor(W1, requires_grad=True)
    tw2 = torch.tensor(W2, requires_grad=True)
    (torch.relu(tx @ tw1) @ tw2).sum().backward()

    x = BasicTensor(X, requires_grad=True)
    w1 = BasicTensor(W1, requires_grad=True)
    w2 = BasicTensor(W2, requires_grad=True)
    ((x @ w1).relu() @ w2).sum().backward()

    assert np.allclose(x.grad, tx.grad.numpy(), atol=1e-4)
    assert np.allclose(w1.grad, tw1.grad.numpy(), atol=1e-4)
    assert np.allclose(w2.grad, tw2.grad.numpy(), atol=1e-4)


# ---------------------------------------------------------------------------
# 17. sum avec axe
# ---------------------------------------------------------------------------

def test_sum_axe_forward():
    a = BasicTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(a.sum(axis=0).data, [5.0, 7.0, 9.0])
    assert np.allclose(a.sum(axis=1).data, [6.0, 15.0])


def test_sum_axe_shape():
    a = BasicTensor(np.random.randn(3, 4, 5))
    assert a.sum(axis=0).data.shape == (4, 5)
    assert a.sum(axis=1).data.shape == (3, 5)
    assert a.sum(axis=2).data.shape == (3, 4)


def test_sum_keepdims_shape():
    a = BasicTensor(np.random.randn(3, 4))
    assert a.sum(axis=1, keepdims=True).data.shape == (3, 1)
    assert a.sum(axis=0, keepdims=True).data.shape == (1, 4)


def test_sum_global_inchange():
    a = BasicTensor([[1.0, 2.0], [3.0, 4.0]])
    assert a.sum().data.shape == ()
    assert np.allclose(a.sum().data, 10.0)


def test_sum_axe_backward_un_partout():
    a = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    a.sum(axis=1).sum().backward()
    assert a.grad.shape == (3, 4)
    assert np.allclose(a.grad, np.ones((3, 4)))


def test_sum_keepdims_backward():
    a = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    a.sum(axis=0, keepdims=True).sum().backward()
    assert a.grad.shape == (3, 4)
    assert np.allclose(a.grad, np.ones((3, 4)))


def test_sum_axe_backward_pondere():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    poids = BasicTensor([10.0, 100.0], requires_grad=False)
    (a.sum(axis=1) * poids).sum().backward()
    assert np.allclose(a.grad[0], 10.0)
    assert np.allclose(a.grad[1], 100.0)


def test_sum_grad_dtype():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    a.sum(axis=1).sum().backward()
    assert a.grad.dtype == np.float32


def test_sum_accumule_deux_fois():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    a.sum(axis=1).sum().backward()
    a.sum(axis=1).sum().backward()
    assert np.allclose(a.grad, 2.0)


def test_gradcheck_sum_axe():
    a = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    gradcheck(lambda x: x.sum(axis=1).sum(), [a])


def test_gradcheck_sum_keepdims():
    a = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    gradcheck(lambda x: x.sum(axis=0, keepdims=True).sum(), [a])


def test_gradcheck_sum_axe_3d():
    a = BasicTensor(np.random.randn(2, 3, 4), requires_grad=True)
    gradcheck(lambda x: x.sum(axis=1).sum(), [a])


def test_sum_contre_pytorch():
    torch = pytest.importorskip("torch")
    donnees = np.random.randn(3, 4).astype(np.float32)

    ta = torch.tensor(donnees, requires_grad=True)
    (ta.sum(dim=1) * ta.sum(dim=1)).sum().backward()

    a = BasicTensor(donnees, requires_grad=True)
    (a.sum(axis=1) * a.sum(axis=1)).sum().backward()

    assert np.allclose(a.grad, ta.grad.numpy(), atol=1e-4)


# ---------------------------------------------------------------------------
# 18. mean
# ---------------------------------------------------------------------------

def test_mean_forward():
    a = BasicTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(a.mean().data, 3.5)
    assert np.allclose(a.mean(axis=1).data, [2.0, 5.0])


def test_mean_backward():
    a = BasicTensor(np.random.randn(2, 5), requires_grad=True)
    a.mean().backward()
    assert np.allclose(a.grad, 1.0 / 10.0, atol=1e-6)


def test_mean_axe_backward():
    a = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    a.mean(axis=1).sum().backward()
    assert np.allclose(a.grad, 1.0 / 4.0, atol=1e-6)


def test_gradcheck_mean():
    a = BasicTensor(np.random.randn(3, 4), requires_grad=True)
    gradcheck(lambda x: x.mean(axis=1).sum(), [a])


# ---------------------------------------------------------------------------
# 19. exp
# ---------------------------------------------------------------------------

def test_exp_forward():
    a = BasicTensor([0.0, 1.0, 2.0])
    assert np.allclose(a.exp().data, [1.0, np.e, np.e ** 2], atol=1e-5)


def test_exp_backward_egale_la_sortie():
    a = BasicTensor([0.0, 1.0, 2.0], requires_grad=True)
    out = a.exp()
    out.sum().backward()
    assert np.allclose(a.grad, out.data, atol=1e-5)


def test_gradcheck_exp():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    gradcheck(lambda x: x.exp().sum(), [a], tol=1e-1, eps=1e-1)


def test_exp_contre_pytorch():
    torch = pytest.importorskip("torch")
    donnees = np.random.randn(2, 3).astype(np.float32)

    ta = torch.tensor(donnees, requires_grad=True)
    ta.exp().sum().backward()

    a = BasicTensor(donnees, requires_grad=True)
    a.exp().sum().backward()

    assert np.allclose(a.grad, ta.grad.numpy(), atol=1e-4)


def test_exp_deborde_sur_grands_logits():
    """exp(800) = inf. -> log-sum-exp trick"""
    a = BasicTensor([800.0])
    assert np.isinf(a.exp().data[0])


# ---------------------------------------------------------------------------
# 20. log
# ---------------------------------------------------------------------------

def test_log_forward():
    a = BasicTensor([1.0, np.e, np.e ** 2])
    assert np.allclose(a.log().data, [0.0, 1.0, 2.0], atol=1e-5)


def test_log_backward_inverse():
    a = BasicTensor([1.0, 2.0, 4.0], requires_grad=True)
    a.log().sum().backward()
    assert np.allclose(a.grad, [1.0, 0.5, 0.25], atol=1e-5)


def test_gradcheck_log():
    a = BasicTensor(np.random.rand(2, 3) + 1.0, requires_grad=True)
    gradcheck(lambda x: x.log().sum(), [a])


def test_log_contre_pytorch():
    torch = pytest.importorskip("torch")
    donnees = (np.random.rand(2, 3) + 1.0).astype(np.float32)

    ta = torch.tensor(donnees, requires_grad=True)
    ta.log().sum().backward()

    a = BasicTensor(donnees, requires_grad=True)
    a.log().sum().backward()

    assert np.allclose(a.grad, ta.grad.numpy(), atol=1e-4)


def test_log_exp_est_identite():
    a = BasicTensor(np.random.rand(2, 3) + 1.0, requires_grad=True)
    assert np.allclose(a.exp().log().data, a.data, atol=1e-4)


def test_gradcheck_log_exp():
    a = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    gradcheck(lambda x: x.exp().log().sum(), [a])


# ---------------------------------------------------------------------------
# 21. Composition softmax naif
# ---------------------------------------------------------------------------

def test_softmax_naif_somme_a_un():
    logits = BasicTensor(np.random.randn(3, 5), requires_grad=True)
    e = logits.exp()
    probas = e / e.sum(axis=1, keepdims=True)
    assert np.allclose(probas.data.sum(axis=1), 1.0, atol=1e-5)


def test_gradcheck_softmax_naif():
    logits = BasicTensor(np.random.randn(3, 4) * 0.5, requires_grad=True)

    def softmax_somme(x):
        e = x.exp()
        return (e / e.sum(axis=1, keepdims=True)).sum()

    gradcheck(softmax_somme, [logits])


def test_softmax_naif_deborde():
    """exp deborde et softmax donne des NaN."""
    logits = BasicTensor([[800.0, 1.0, 2.0]])
    e = logits.exp()
    probas = e / e.sum(axis=1, keepdims=True)
    assert np.isnan(probas.data).any() or np.isinf(probas.data).any()

# ---------------------------------------------------------------------------
# 22. naive_softmax
# ---------------------------------------------------------------------------

def one_hot(indices, nb_classes):
    """Helper de test : indices (N,) -> one-hot (N, C) en BasicTensor."""
    indices = np.asarray(indices)
    m = np.zeros((len(indices), nb_classes), dtype=np.float32)
    m[np.arange(len(indices)), indices] = 1.0
    return BasicTensor(m)


def test_softmax_valeurs_connues():
    z = BasicTensor([[2.0, 1.0, 0.0]])
    p = z.naive_softmax()
    assert np.allclose(p.data, [[0.665, 0.245, 0.090]], atol=1e-3)


def test_softmax_chaque_ligne_somme_a_un():
    z = BasicTensor(np.random.randn(5, 4))
    p = z.naive_softmax()
    assert np.allclose(p.data.sum(axis=-1), 1.0, atol=1e-5)


def test_softmax_shape():
    z = BasicTensor(np.random.randn(5, 4))
    assert z.naive_softmax().data.shape == (5, 4)


def test_softmax_positif():
    z = BasicTensor(np.random.randn(5, 4) * 3)
    assert (z.naive_softmax().data > 0).all()


def test_softmax_logits_egaux_donne_uniforme():
    z = BasicTensor(np.zeros((2, 4)))
    assert np.allclose(z.naive_softmax().data, 0.25)


def test_softmax_invariant_par_translation():
    z = np.random.randn(3, 4).astype(np.float32)
    p1 = BasicTensor(z).naive_softmax().data
    p2 = BasicTensor(z + 5.0).naive_softmax().data
    assert np.allclose(p1, p2, atol=1e-5)


def test_softmax_1d():
    z = BasicTensor([2.0, 1.0, 0.0])
    p = z.naive_softmax()
    assert np.isclose(p.data.sum(), 1.0, atol=1e-5)


def test_softmax_contre_pytorch():
    torch = pytest.importorskip("torch")
    z = np.random.randn(3, 4).astype(np.float32)
    ref = torch.softmax(torch.tensor(z), dim=-1).numpy()
    assert np.allclose(BasicTensor(z).naive_softmax().data, ref, atol=1e-5)


def test_gradcheck_softmax():
    z = BasicTensor(np.random.randn(3, 4) * 0.5, requires_grad=True)
    poids = BasicTensor(np.random.randn(3, 4))
    gradcheck(lambda x: (x.naive_softmax() * poids).sum(), [z])


# ---------------------------------------------------------------------------
# 23. naive_cross_entropy
# ---------------------------------------------------------------------------

def test_ce_valeur_connue():
    """z = [2, 1, 0], bonne classe 0 -> -log(0.665) = 0.408"""
    z = BasicTensor([[2.0, 1.0, 0.0]])
    loss = z.naive_softmax().naive_cross_entropy(one_hot([0], 3))
    assert loss.data.shape == ()
    assert np.isclose(loss.data, 0.408, atol=1e-3)


def test_ce_renvoie_un_scalaire():
    z = BasicTensor(np.random.randn(8, 5))
    loss = z.naive_softmax().naive_cross_entropy(one_hot(np.random.randint(0, 5, 8), 5))
    assert loss.data.shape == ()


def test_ce_logits_uniformes_donne_log_c():
    z = BasicTensor(np.zeros((4, 10)))
    loss = z.naive_softmax().naive_cross_entropy(one_hot([0, 3, 7, 9], 10))
    assert np.isclose(loss.data, np.log(10), atol=1e-4)


def test_ce_bonne_prediction_loss_faible():
    z = BasicTensor([[10.0, 0.0, 0.0]])
    loss = z.naive_softmax().naive_cross_entropy(one_hot([0], 3))
    assert loss.data < 0.01


def test_ce_mauvaise_prediction_loss_forte():
    z = BasicTensor([[10.0, 0.0, 0.0]])
    loss = z.naive_softmax().naive_cross_entropy(one_hot([2], 3))
    assert loss.data > 5.0


def test_ce_toujours_positive():
    z = BasicTensor(np.random.randn(6, 4))
    loss = z.naive_softmax().naive_cross_entropy(one_hot(np.random.randint(0, 4, 6), 4))
    assert loss.data >= 0


def test_ce_moyenne_independante_de_la_taille_du_batch():
    z = np.random.randn(3, 4).astype(np.float32)
    cibles = [0, 2, 1]
    l1 = BasicTensor(z).naive_softmax().naive_cross_entropy(one_hot(cibles, 4))
    l2 = BasicTensor(np.concatenate([z, z])).naive_softmax().naive_cross_entropy(
        one_hot(cibles + cibles, 4))
    assert np.isclose(l1.data, l2.data, atol=1e-5)


def test_ce_accepte_soft_target():
    """La forme generale -sum(y * log p) marche aussi avec des cibles non one-hot."""
    z = BasicTensor([[2.0, 1.0, 0.0]])
    cible = BasicTensor([[0.5, 0.5, 0.0]])
    loss = z.naive_softmax().naive_cross_entropy(cible)
    attendu = -(0.5 * np.log(0.665) + 0.5 * np.log(0.245))
    assert np.isclose(loss.data, attendu, atol=1e-2)


def test_ce_contre_pytorch():
    torch = pytest.importorskip("torch")
    z = np.random.randn(5, 4).astype(np.float32)
    cibles = np.random.randint(0, 4, 5)

    ref = torch.nn.functional.cross_entropy(torch.tensor(z), torch.tensor(cibles, dtype=torch.long))

    loss = BasicTensor(z).naive_softmax().naive_cross_entropy(one_hot(cibles, 4))
    assert np.isclose(loss.data, ref.item(), atol=1e-5)


# ---------------------------------------------------------------------------
# 24. Gradients de softmax + CE
# ---------------------------------------------------------------------------

def test_gradcheck_ce():
    z = BasicTensor(np.random.randn(4, 3) * 0.5, requires_grad=True)
    cible = one_hot([0, 2, 1, 1], 3)
    gradcheck(lambda x: x.naive_softmax().naive_cross_entropy(cible), [z])


def test_ce_gradient_vaut_p_moins_y_sur_n():
    N, C = 4, 3
    z = BasicTensor(np.random.randn(N, C), requires_grad=True)
    cible = one_hot([0, 2, 1, 1], C)

    p = z.naive_softmax()
    p.naive_cross_entropy(cible).backward()

    attendu = (p.data - cible.data) / N
    assert np.allclose(z.grad, attendu, atol=1e-5)


def test_ce_gradient_contre_pytorch():
    torch = pytest.importorskip("torch")
    z = np.random.randn(5, 4).astype(np.float32)
    cibles = np.random.randint(0, 4, 5)

    tz = torch.tensor(z, requires_grad=True)
    torch.nn.functional.cross_entropy(tz, torch.tensor(cibles, dtype=torch.long)).backward()

    bz = BasicTensor(z, requires_grad=True)
    bz.naive_softmax().naive_cross_entropy(one_hot(cibles, 4)).backward()

    assert np.allclose(bz.grad, tz.grad.numpy(), atol=1e-5)


def test_ce_cible_sans_gradient():
    z = BasicTensor(np.random.randn(2, 3), requires_grad=True)
    cible = one_hot([0, 1], 3)
    z.naive_softmax().naive_cross_entropy(cible).backward()
    assert cible.grad is None


# ---------------------------------------------------------------------------
# 25. Version naive casse sur de grands logits
#
# Ces tests permettent de voir pourquoi on utilise une autre formule pour la CE
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_softmax_naif_casse_sur_grands_logits():
    z = BasicTensor([[1000.0, 0.0, 0.0]])
    p = z.naive_softmax()
    assert np.isnan(p.data).any()       # exp(1000) = inf, inf / inf = nan


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_ce_naive_casse_sur_grands_logits():
    z = BasicTensor([[1000.0, 0.0, 0.0]])
    loss = z.naive_softmax().naive_cross_entropy(one_hot([0], 3))
    assert not np.isfinite(loss.data)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_ce_naive_casse_sur_logits_tres_negatifs():
    """exp(-1000) = 0, proba devient 0, log(0) = -inf."""
    z = BasicTensor([[0.0, 1000.0, 0.0]])
    loss = z.naive_softmax().naive_cross_entropy(one_hot([0], 3))
    assert not np.isfinite(loss.data)


def test_pytorch_ne_casse_pas_lui():
    """Meme entree, PyTorch donne un resultat fini"""
    torch = pytest.importorskip("torch")
    ref = torch.nn.functional.cross_entropy(
        torch.tensor([[1000.0, 0.0, 0.0]]), torch.tensor([0]))
    assert np.isfinite(ref.item())


def test_softmax_stable_sur_grands_logits():
    z = BasicTensor([[1000.0, 0.0, 0.0]])
    p = z.stable_softmax()
    # Ne casse plus !
    assert not np.isnan(p.data).any()