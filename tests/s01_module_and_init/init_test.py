

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.s00_autograd_and_tensors.autograd import BasicTensor, no_grad
from src.s01_module_and_init.init import Init
from src.s01_module_and_init.module import BasicModule, BasicParameter
from src.s01_module_and_init.layers import Linear

# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------

def ks_statistique(a, b):
    """
    Test de Kolmogorov-Smirnov : ecart max
    entre les deux fonctions de repartition empiriques.
    Doit etre proche de 0 si c'est la meme distrib
    """
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    points = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, points, side="right") / a.size
    cdf_b = np.searchsorted(b, points, side="right") / b.size
    return np.abs(cdf_a - cdf_b).max()


def ks_seuil(n, m, alpha=0.001):
    """Seuil test KS"""
    c = np.sqrt(-0.5 * np.log(alpha / 2))
    return c * np.sqrt((n + m) / (n * m))


def assert_meme_distribution(nos_poids, poids_torch):
    d = ks_statistique(nos_poids, poids_torch)
    seuil = ks_seuil(nos_poids.size, poids_torch.size)
    assert d < seuil, f"distributions differentes : KS = {d:.5f} > seuil {seuil:.5f}"


def nos_poids(init, shape, **kwargs):
    p = BasicParameter(np.zeros(shape))
    init(p, np.random.default_rng(0), **kwargs)
    return p.data


def poids_torch(init_torch, shape, **kwargs):
    torch.manual_seed(0)
    t = torch.empty(shape)
    init_torch(t, **kwargs)
    return t.numpy()


SHAPES = [
    (256, 1000),        # Linear(1000, 256)
    (64, 32, 3, 3),     # Conv2d(32, 64, 3)
]


# ---------------------------------------------------------------------------
# 1. Fans
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (128, 784),         # Linear
    (10, 1),            # Linear a une entree
    (16, 3, 5),         # Conv1d
    (16, 3, 5, 5),      # Conv2d
    (8, 4, 3, 3, 3),    # Conv3d
])
def test_fans_identiques_a_pytorch(shape):
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(shape))
    assert Init._fan_in(shape) == fan_in
    assert Init._fan_out(shape) == fan_out


# ---------------------------------------------------------------------------
# 2. Distributions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
def test_he_normal_comme_kaiming_normal(shape):
    assert_meme_distribution(
        nos_poids(Init.he_, shape, distribution="normal"),
        poids_torch(torch.nn.init.kaiming_normal_, shape, mode="fan_in", nonlinearity="relu"),
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_he_uniform_comme_kaiming_uniform(shape):
    assert_meme_distribution(
        nos_poids(Init.he_, shape, distribution="uniform"),
        poids_torch(torch.nn.init.kaiming_uniform_, shape, mode="fan_in", nonlinearity="relu"),
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_xavier_normal_comme_pytorch(shape):
    assert_meme_distribution(
        nos_poids(Init.xavier_, shape, distribution="normal"),
        poids_torch(torch.nn.init.xavier_normal_, shape),
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_xavier_uniform_comme_pytorch(shape):
    assert_meme_distribution(
        nos_poids(Init.xavier_, shape, distribution="uniform"),
        poids_torch(torch.nn.init.xavier_uniform_, shape),
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_lecun_normal_comme_kaiming_lineaire(shape):
    """Pas de LeCun coté pytorch -> kaiming_normal_ avec gain 1 (nonlinearity='linear')."""
    assert_meme_distribution(
        nos_poids(Init.lecun_, shape, distribution="normal"),
        poids_torch(torch.nn.init.kaiming_normal_, shape, mode="fan_in", nonlinearity="linear"),
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_normale_tronquee_comme_trunc_normal(shape):
    """
    torch.nn.init.trunc_normal_ coupe a des bornes absolues
    Init doit correspondre a une troncature a +/- 2 ecarts-types de l'ecart-type
    corrige + avoir au final la variance visee (2 / fan_in pour He).
    """
    var_visee = 2 / Init._fan_in(shape)
    std_avant_troncature = np.sqrt(var_visee) / 0.87962566103423978

    nous = nos_poids(Init.he_, shape, distribution="truncated_normal")
    eux = poids_torch(torch.nn.init.trunc_normal_, shape, mean=0.0, std=std_avant_troncature,
                      a=-2 * std_avant_troncature, b=2 * std_avant_troncature)

    assert_meme_distribution(nous, eux)
    assert np.isclose(nous.var(), var_visee, rtol=0.05)


@pytest.mark.parametrize("shape", SHAPES)
def test_normal_comme_pytorch(shape):
    assert_meme_distribution(
        nos_poids(Init.normal_, shape, std=0.02),
        poids_torch(torch.nn.init.normal_, shape, mean=0.0, std=0.02),
    )


def test_ks_detecte_une_mauvaise_echelle():
    """KS doit echouer si l'echelle est fausse d'un facteur 2."""
    shape = (256, 1000)
    nous = nos_poids(Init.he_, shape, distribution="normal")
    eux = poids_torch(torch.nn.init.kaiming_normal_, shape, mode="fan_in", nonlinearity="relu")
    with pytest.raises(AssertionError):
        assert_meme_distribution(nous * np.sqrt(2), eux)


# ---------------------------------------------------------------------------
# 3. Linear : meme forward et memes gradients que torch.nn.Linear
# ---------------------------------------------------------------------------

def lineaires_jumeaux(n_in, n_out, bias=True):
    """torch.nn.Linear et notre Linear avec exactement les memes poids."""
    torch.manual_seed(0)
    eux = torch.nn.Linear(n_in, n_out, bias=bias)
    nous = Linear(n_in, n_out, bias=bias)
    nous.weight.data = eux.weight.detach().numpy().copy()
    if bias:
        nous.bias.data = eux.bias.detach().numpy().copy()
    return nous, eux


def comparer_forward_backward(nous, eux, shape_entree):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(shape_entree).astype(np.float32)

    bx = BasicTensor(x, requires_grad=True)
    tx = torch.tensor(x, requires_grad=True)

    b_out = nous(bx)
    t_out = eux(tx)
    assert b_out.shape == tuple(t_out.shape)
    assert np.allclose(b_out.data, t_out.detach().numpy(), atol=1e-5)

    # Ponderation aleatoire de la sortie : sinon tous les gradients amont valent 1
    w = rng.standard_normal(t_out.shape).astype(np.float32)
    (b_out * BasicTensor(w)).sum().backward()
    (t_out * torch.tensor(w)).sum().backward()

    assert np.allclose(nous.weight.grad, eux.weight.grad.numpy(), atol=1e-4)
    assert np.allclose(bx.grad, tx.grad.numpy(), atol=1e-4)
    if eux.bias is not None:
        assert np.allclose(nous.bias.grad, eux.bias.grad.numpy(), atol=1e-4)


def test_linear_batch_comme_pytorch():
    nous, eux = lineaires_jumeaux(5, 3)
    comparer_forward_backward(nous, eux, (8, 5))


def test_linear_sans_biais_comme_pytorch():
    nous, eux = lineaires_jumeaux(5, 3, bias=False)
    comparer_forward_backward(nous, eux, (8, 5))
    assert len(list(nous.parameters())) == len(list(eux.parameters())) == 1


def test_linear_entree_3d_comme_pytorch():
    """(batch, sequence, features) -> gradient de W est la somme sur 2 axes."""
    nous, eux = lineaires_jumeaux(5, 3)
    comparer_forward_backward(nous, eux, (2, 4, 5))


def test_linear_entree_1d_comme_pytorch():
    nous, eux = lineaires_jumeaux(5, 3)
    comparer_forward_backward(nous, eux, (5,))


def test_linear_memes_shapes_de_parametres_que_pytorch():
    nous, eux = lineaires_jumeaux(7, 4)
    assert [p.shape for p in nous.parameters()] == [tuple(p.shape) for p in eux.parameters()]


# ---------------------------------------------------------------------------
# 4. BasicModule
# ---------------------------------------------------------------------------

class BasicMLP(BasicModule):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.fc2 = Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())


class TorchMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def mlps_jumeaux():
    torch.manual_seed(0)
    eux = TorchMLP()
    nous = BasicMLP()
    for p_nous, p_eux in zip(nous.parameters(), eux.parameters()):
        p_nous.data = p_eux.detach().numpy().copy()
    return nous, eux


def test_parameters_meme_ordre_et_memes_shapes_que_pytorch():
    nous, eux = BasicMLP(), TorchMLP()
    assert [p.shape for p in nous.parameters()] == [tuple(p.shape) for p in eux.parameters()]


def test_poids_partage_compte_une_fois_comme_pytorch():
    class NotreTie(BasicModule):
        def __init__(self):
            super().__init__()
            self.a = Linear(4, 4)
            self.b = Linear(4, 4)
            self.b.weight = self.a.weight

    class LeurTie(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(4, 4)
            self.b = torch.nn.Linear(4, 4)
            self.b.weight = self.a.weight

    assert len(list(NotreTie().parameters())) == len(list(LeurTie().parameters())) == 3


def test_poids_partage_recoit_la_somme_des_gradients_comme_pytorch():
    torch.manual_seed(0)
    eux = torch.nn.Linear(4, 4)
    nous = Linear(4, 4)
    nous.weight.data = eux.weight.detach().numpy().copy()
    nous.bias.data = eux.bias.detach().numpy().copy()

    x = np.random.default_rng(2).standard_normal((3, 4)).astype(np.float32)
    nous(nous(BasicTensor(x))).sum().backward()      # la meme couche utilisee deux fois
    eux(eux(torch.tensor(x))).sum().backward()

    assert np.allclose(nous.weight.grad, eux.weight.grad.numpy(), atol=1e-4)
    assert np.allclose(nous.bias.grad, eux.bias.grad.numpy(), atol=1e-4)


def test_zero_grad_comme_pytorch():
    """PyTorch remet les gradients a None par defaut (set_to_none=True)."""
    nous, eux = mlps_jumeaux()
    x = np.random.default_rng(3).standard_normal((5, 4)).astype(np.float32)
    nous(BasicTensor(x)).sum().backward()
    eux(torch.tensor(x)).sum().backward()
    nous.zero_grad()
    eux.zero_grad()
    assert all(p.grad is None for p in nous.parameters())
    assert all(p.grad is None for p in eux.parameters())


def test_mlp_forward_backward_comme_pytorch():
    nous, eux = mlps_jumeaux()
    x = np.random.default_rng(4).standard_normal((6, 4)).astype(np.float32)

    b_out = nous(BasicTensor(x))
    t_out = eux(torch.tensor(x))
    assert np.allclose(b_out.data, t_out.detach().numpy(), atol=1e-5)

    b_out.sum().backward()
    t_out.sum().backward()
    for p_nous, p_eux in zip(nous.parameters(), eux.parameters()):
        assert np.allclose(p_nous.grad, p_eux.grad.numpy(), atol=1e-4)


# ---------------------------------------------------------------------------
# 5. Entrainement
# ---------------------------------------------------------------------------

def basic_sgd(modele, lr):
    with no_grad():
        for p in modele.parameters():
            p.data -= lr * p.grad
    modele.zero_grad()


def torch_sgd(modele, lr):
    with torch.no_grad():
        for p in modele.parameters():
            p -= lr * p.grad
    modele.zero_grad()


def test_regression_mse_meme_trajectoire_que_pytorch():
    torch.manual_seed(0)
    eux = torch.nn.Linear(3, 1)
    nous = Linear(3, 1)
    nous.weight.data = eux.weight.detach().numpy().copy()
    nous.bias.data = eux.bias.detach().numpy().copy()

    rng = np.random.default_rng(5)
    x = rng.standard_normal((32, 3)).astype(np.float32)
    y = (x @ np.array([[2.0], [-1.0], [0.5]], dtype=np.float32) + 0.3).astype(np.float32)

    pertes_nous, pertes_eux = [], []
    for _ in range(20):
        l_nous = ((nous(BasicTensor(x)) - BasicTensor(y)) ** 2).mean()
        l_eux = ((eux(torch.tensor(x)) - torch.tensor(y)) ** 2).mean()
        l_nous.backward()
        l_eux.backward()
        pertes_nous.append(float(l_nous.data))
        pertes_eux.append(l_eux.item())
        basic_sgd(nous, lr=0.1)
        torch_sgd(eux, lr=0.1)

    assert np.allclose(pertes_nous, pertes_eux, rtol=1e-4, atol=1e-5)
    assert np.allclose(nous.weight.data, eux.weight.detach().numpy(), atol=1e-4)
    assert pertes_nous[-1] < pertes_nous[0]


def test_classification_cross_entropy_meme_trajectoire_que_pytorch():
    nous, eux = mlps_jumeaux()

    rng = np.random.default_rng(6)
    x = rng.standard_normal((16, 4)).astype(np.float32)
    cibles = rng.integers(0, 3, 16)
    one_hot = np.eye(3, dtype=np.float32)[cibles]

    pertes_nous, pertes_eux = [], []
    for _ in range(20):
        l_nous = nous(BasicTensor(x)).cross_entropy(BasicTensor(one_hot))
        l_eux = torch.nn.functional.cross_entropy(eux(torch.tensor(x)),
                                                  torch.tensor(cibles, dtype=torch.long))
        l_nous.backward()
        l_eux.backward()
        pertes_nous.append(float(l_nous.data))
        pertes_eux.append(l_eux.item())
        basic_sgd(nous, lr=0.5)
        torch_sgd(eux, lr=0.5)

    assert np.allclose(pertes_nous, pertes_eux, rtol=1e-4, atol=1e-5)
    for p_nous, p_eux in zip(nous.parameters(), eux.parameters()):
        assert np.allclose(p_nous.data, p_eux.detach().numpy(), atol=1e-4)
    assert pertes_nous[-1] < pertes_nous[0]