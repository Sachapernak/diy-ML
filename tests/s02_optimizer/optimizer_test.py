import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.s00_autograd_and_tensors.autograd import BasicTensor, default_dtype
from src.s01_module_and_init.layers import Sequential, Linear, ReLU, BasicParameter
from src.s02_optimizer.optimizer import SGD, MomentumGradient, Adam, AdamW, AdEMAMix
NB_PAS = 30


# ---------------------------------------------------------------------------
# Les optimiseurs a comparer
# ---------------------------------------------------------------------------

# Pour chaque cas : optim vs ceux de PyTorch avec les memes hyperparametres.
CAS = {
    "sgd": (
        lambda p: SGD(p, learning_rate=0.1),
        lambda p: torch.optim.SGD(p, lr=0.1),
    ),
    "momentum": (
        lambda p: MomentumGradient(p, learning_rate=0.05, momentum=0.9),
        lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
    ),
    "momentum_dampening": (
        lambda p: MomentumGradient(p, learning_rate=0.05, momentum=0.9, dampening=0.5),
        lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, dampening=0.5),
    ),
    "adam": (
        lambda p: Adam(p, learning_rate=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, decay=0.0),
        lambda p: torch.optim.Adam(p, lr=1e-2, betas=(0.9, 0.999), eps=1e-8),
    ),
    # betas inhabituels
    "adam_autres_betas": (
        lambda p: Adam(p, learning_rate=1e-2, beta1=0.8, beta2=0.99, eps=1e-8, decay=0.0),
        lambda p: torch.optim.Adam(p, lr=1e-2, betas=(0.8, 0.99), eps=1e-8),
    ),
    # weight_decay d'Adam
    "adam_weight_decay": (
        lambda p: Adam(p, learning_rate=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, decay=0.1),
        lambda p: torch.optim.Adam(p, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1),
    ),
    "adamw": (
        lambda p: AdamW(p, 1e-2, 0.9, 0.999, 1e-8, 0.1),
        lambda p: torch.optim.AdamW(p, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1),
    ),
}


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------

def mlps_jumeaux():
    """Notre MLP et celui de PyTorch, les memes poids (float64)."""
    torch.manual_seed(0)
    eux = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)).double()
    nous = Sequential(Linear(4, 8), ReLU(), Linear(8, 3))
    for p_nous, p_eux in zip(nous.parameters(), eux.parameters()):
        p_nous.data = p_eux.detach().numpy().copy()
    return nous, eux


def lineaires_jumeaux(seed):
    torch.manual_seed(seed)
    eux = torch.nn.Linear(4, 3).double()
    nous = Linear(4, 3)
    nous.weight.data = eux.weight.detach().numpy().copy()
    nous.bias.data = eux.bias.detach().numpy().copy()
    return nous, eux


def minibatch(seed, n=16):
    """Un minibatch different a chaque pas"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 4))
    cibles = rng.integers(0, 3, n)
    return x, cibles, np.eye(3)[cibles]


def loss_nous(modele, x, one_hot):
    return modele(BasicTensor(x)).cross_entropy(BasicTensor(one_hot))


def loss_eux(modele, x, cibles):
    return torch.nn.functional.cross_entropy(modele(torch.tensor(x)),
                                             torch.tensor(cibles, dtype=torch.long))


def assert_memes_parametres(params_nous, params_eux, pas):
    for i, (p_nous, p_eux) in enumerate(zip(params_nous, params_eux)):
        np.testing.assert_allclose(
            p_nous.data, p_eux.detach().numpy(), rtol=1e-6, atol=1e-8,
            err_msg=f"parametre {i}, apres le pas {pas}",
        )


# ---------------------------------------------------------------------------
# 1. Entrainement normal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cas", CAS.keys())
def test_meme_trajectoire_que_pytorch(cas):
    fabrique_nous, fabrique_eux = CAS[cas]

    with default_dtype(np.float64):
        nous, eux = mlps_jumeaux()
        opt_nous = fabrique_nous(nous.parameters())
        opt_eux = fabrique_eux(eux.parameters())

        for pas in range(NB_PAS):
            x, cibles, one_hot = minibatch(seed=pas)

            opt_nous.zero_grad()
            opt_eux.zero_grad()
            loss_nous(nous, x, one_hot).backward()
            loss_eux(eux, x, cibles).backward()
            opt_nous.step()
            opt_eux.step()

            assert_memes_parametres(nous.parameters(), eux.parameters(), pas)


# ---------------------------------------------------------------------------
# 2. Un parametre sans gradient a certains pas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cas", CAS.keys())
def test_parametre_sans_gradient_un_pas_sur_deux(cas):
    """
    Deux couches independantes avec la 2nd qui ne sert qu'un pas sur deux, et n'a donc
    pas de gradient les autres pas.

    PyTorch l'ignore completement

    """
    fabrique_nous, fabrique_eux = CAS[cas]

    with default_dtype(np.float64):
        a_nous, a_eux = lineaires_jumeaux(seed=0)
        b_nous, b_eux = lineaires_jumeaux(seed=1)
        params_nous = list(a_nous.parameters()) + list(b_nous.parameters())
        params_eux = list(a_eux.parameters()) + list(b_eux.parameters())
        opt_nous = fabrique_nous(params_nous)
        opt_eux = fabrique_eux(params_eux)

        for pas in range(NB_PAS):
            x, cibles, one_hot = minibatch(seed=pas)

            l_nous = loss_nous(a_nous, x, one_hot)
            l_eux = loss_eux(a_eux, x, cibles)
            if pas % 2 == 0:
                l_nous = l_nous + loss_nous(b_nous, x, one_hot)
                l_eux = l_eux + loss_eux(b_eux, x, cibles)

            opt_nous.zero_grad()
            opt_eux.zero_grad()
            l_nous.backward()
            l_eux.backward()
            opt_nous.step()
            opt_eux.step()

            assert_memes_parametres(params_nous, params_eux, pas)


# ---------------------------------------------------------------------------
# 3. Sans zero_grad
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cas", CAS.keys())
def test_sans_zero_grad_meme_trajectoire_que_pytorch(cas):
    """
    oublie de zero_grad expres -> chaque backward ajoute son gradient a l'ancien
    """
    fabrique_nous, fabrique_eux = CAS[cas]

    with default_dtype(np.float64):
        nous, eux = mlps_jumeaux()
        opt_nous = fabrique_nous(nous.parameters())
        opt_eux = fabrique_eux(eux.parameters())

        for pas in range(10):
            x, cibles, one_hot = minibatch(seed=pas)

            loss_nous(nous, x, one_hot).backward()
            loss_eux(eux, x, cibles).backward()
            opt_nous.step()
            opt_eux.step()

            assert_memes_parametres(nous.parameters(), eux.parameters(), pas)


# ---------------------------------------------------------------------------
# 4. Adam + L2 et AdamW
# ---------------------------------------------------------------------------

def trajectoire(fabrique, nb_pas=NB_PAS):
    """Entraine notre MLP avec l'optimiseur et renvoie les parametres."""
    nous, _ = mlps_jumeaux()
    opt = fabrique(nous.parameters())
    for pas in range(nb_pas):
        x, _, one_hot = minibatch(seed=pas)
        opt.zero_grad()
        loss_nous(nous, x, one_hot).backward()
        opt.step()
    return [p.data.copy() for p in nous.parameters()]


def test_adamw_sans_decay_est_adam():
    """Avec weight_decay = 0 -> pas de diff entre adam et adamW."""
    with default_dtype(np.float64):
        adam = trajectoire(lambda p: Adam(p, 1e-2, 0.9, 0.999, 1e-8, 0.0))
        adamw = trajectoire(lambda p: AdamW(p, 1e-2, 0.9, 0.999, 1e-8, 0.0))

    for a, b in zip(adam, adamw):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_adam_l2_et_adamw_divergent():
    """
    Resultats different entre adamW et adam, malgré les meme paramètres
    avec Adam, wd * theta passe par m et est divise par sqrt(v) ;
    avec AdamW, il s'applique directement aux poids.
    Avec SGD, les deux seraient equivalents (voir le test suivant).
    """
    with default_dtype(np.float64):
        adam_l2 = trajectoire(lambda p: Adam(p, 1e-2, 0.9, 0.999, 1e-8, 0.1))
        adamw = trajectoire(lambda p: AdamW(p, 1e-2, 0.9, 0.999, 1e-8, 0.1))

    ecart = max(np.abs(a - b).max() for a, b in zip(adam_l2, adamw))
    assert ecart > 1e-3


def test_avec_sgd_l2_et_decay_decouple_sont_identiques():
    """
    Avec un sgd le weight decay directement sur les poids ou sur le gradient donne la meme chose, au contraire d'adam et
    d'adamW
    """
    torch.manual_seed(0)
    a = torch.nn.Linear(4, 3).double()
    b = torch.nn.Linear(4, 3).double()
    b.load_state_dict(a.state_dict())

    lr, wd = 0.1, 0.1
    opt_l2 = torch.optim.SGD(a.parameters(), lr=lr, weight_decay=wd)
    opt_plain = torch.optim.SGD(b.parameters(), lr=lr)

    for pas in range(NB_PAS):
        x, cibles, _ = minibatch(seed=pas)
        for modele, opt in ((a, opt_l2), (b, opt_plain)):
            opt.zero_grad()
            loss_eux(modele, x, cibles).backward()
            if opt is opt_plain:
                with torch.no_grad():
                    for p in modele.parameters():
                        p.mul_(1 - lr * wd)  # decroissance decouplee, a la main
            opt.step()

    for p_a, p_b in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p_a, p_b, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. AdEMAMix
# ---------------------------------------------------------------------------

X0_ROSENBROCK = np.array([-3.0, 5.0])
HYPER_ADEMAMIX = dict(lr=1e-2, beta1=0.9, beta2=0.999, beta3=0.999, alpha=8.0, eps=1e-8, wd=0.01)


def rosenbrock(theta):
    x1, x2 = theta[0], theta[1]
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2


def ademamix_reference(lr, beta1, beta2, beta3, alpha, eps, wd, t_alpha=None, t_beta=None, nb_pas=300):
    """L'Algorithm 1 du papier avec le gradient analytique de Rosenbrock."""
    x = X0_ROSENBROCK.copy()
    m1, m2, v = np.zeros(2), np.zeros(2), np.zeros(2)
    trajectoire = []
    for t in range(1, nb_pas + 1):
        x1, x2 = x
        g = np.array([-2 * (1 - x1) - 400 * x1 * (x2 - x1 ** 2), 200 * (x2 - x1 ** 2)])

        a = alpha if t_alpha is None else min(t * alpha / t_alpha, alpha)
        b3 = beta3
        if t_beta is not None:
            ln_s, ln_3 = np.log(beta1), np.log(beta3)
            b3 = min(np.exp(ln_s * ln_3 / ((1 - t / t_beta) * ln_3 + (t / t_beta) * ln_s)), beta3)

        m1 = beta1 * m1 + (1 - beta1) * g
        m2 = b3 * m2 + (1 - b3) * g  # pas de correction de biais sur m2
        v = beta2 * v + (1 - beta2) * g ** 2
        m1_hat, v_hat = m1 / (1 - beta1 ** t), v / (1 - beta2 ** t)

        x = x - lr * ((m1_hat + a * m2) / (np.sqrt(v_hat) + eps) + wd * x)
        trajectoire.append(x.copy())
    return np.array(trajectoire)


def trajectoire_rosenbrock(fabrique, nb_pas=300):
    with default_dtype(np.float64):
        theta = BasicParameter(X0_ROSENBROCK.copy())
        opt = fabrique([theta])
        trajectoire = []
        for _ in range(nb_pas):
            opt.zero_grad()
            rosenbrock(theta).backward()
            opt.step()
            trajectoire.append(theta.data.copy())
    return np.array(trajectoire)


def notre_ademamix(lr, beta1, beta2, beta3, alpha, eps, wd, t_alpha=None, t_beta=None):
    return lambda p: AdEMAMix(p, learning_rate=lr, beta1=beta1, beta2=beta2, eps=eps, decay=wd,
                              beta3=beta3, alpha=alpha, t_alpha=t_alpha, t_beta=t_beta)


@pytest.mark.parametrize("schedulers", [
    {},  # sans scheduler
    {"t_alpha": 200, "t_beta": 200},  # montee sur 200 pas, puis plateau jusqu'a 300
])
def test_ademamix_meme_trajectoire_que_la_reference(schedulers):
    hyper = dict(HYPER_ADEMAMIX, **schedulers)
    nous = trajectoire_rosenbrock(notre_ademamix(**hyper))
    reference = ademamix_reference(**hyper)
    np.testing.assert_allclose(nous, reference, rtol=1e-7, atol=1e-10)


def test_ademamix_alpha_nul_redonne_adamw():
    """m2 ne vaut plus rien -> on revient sur AdamW."""
    h = HYPER_ADEMAMIX
    nous = trajectoire_rosenbrock(notre_ademamix(**dict(h, alpha=0.0)))
    adamw = trajectoire_rosenbrock(lambda p: AdamW(p, h["lr"], h["beta1"], h["beta2"], h["eps"], h["wd"]))
    np.testing.assert_allclose(nous, adamw, rtol=1e-12, atol=1e-14)


def test_ademamix_demi_vie_de_m2_croit_lineairement():
    """
    Le scheduler de beta3 fait croitre lineairement la demi-vie de m2, ln(0.5) / ln(beta3(t))
    """
    with default_dtype(np.float64):
        opt = AdEMAMix([BasicParameter(np.zeros(2))])
    T = 1000
    betas = np.array([opt._fbeta3(t, 0.9999, 0.9, T) for t in np.linspace(0, T, 11)])
    demi_vies = np.log(0.5) / np.log(betas)

    assert np.isclose(betas[0], 0.9) and np.isclose(betas[-1], 0.9999)
    np.testing.assert_allclose(demi_vies, np.linspace(demi_vies[0], demi_vies[-1], 11), rtol=1e-6)
    assert betas[1] > 0.998