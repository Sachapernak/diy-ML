"""
Tests pour BasicModule et BasicParameter.

"""

import numpy as np
import pytest


from src.s01_module_and_init.module import BasicModule, BasicParameter, BasicTensor


# ---------------------------------------------------------------------------
# Modules fictifs
# ---------------------------------------------------------------------------

class Feuille(BasicModule):
    """Exemple de couche lineaire avec deux parametres et pas d'enfant."""
    def __init__(self, n_in=2, n_out=3):
        super().__init__()
        self.W = BasicParameter(np.random.randn(n_in, n_out))
        self.b = BasicParameter(np.zeros(n_out))

    def forward(self, x):
        return x @ self.W + self.b


class Conteneur(BasicModule):
    """Un 'MLP' avec deux couches, deux enfants et aucun parametre propre."""
    def __init__(self):
        super().__init__()
        self.fc1 = Feuille(2, 3)
        self.fc2 = Feuille(3, 1)

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())


class Profond(BasicModule):
    """Module avec des sous-modules Profond -> bloc (Conteneur) -> fc1, fc2 (Feuille) et un parametre propre."""
    def __init__(self):
        super().__init__()
        self.echelle = BasicParameter(np.ones(1))
        self.bloc = Conteneur()

    def forward(self, x):
        return self.bloc(x) * self.echelle


class SansForward(BasicModule):
    pass


class AvecArguments(BasicModule):
    def forward(self, a, b=0, *, c=0):
        return (a, b, c)


# ---------------------------------------------------------------------------
# 1. BasicParameter
# ---------------------------------------------------------------------------

def test_parameter_requires_grad_par_defaut():
    assert BasicParameter(np.zeros(3)).requires_grad


def test_parameter_est_un_tenseur():
    p = BasicParameter([1.0, 2.0])
    assert isinstance(p, BasicTensor)
    assert np.allclose((p * 2.0).data, [2.0, 4.0])


def test_op_sur_parameter_donne_un_tenseur_ordinaire():
    """Une activation n'est pas un parametre."""
    p = BasicParameter([1.0, 2.0])
    out = p * 2.0
    assert not isinstance(out, BasicParameter)


# ---------------------------------------------------------------------------
# 2. Construction et enregistrement automatique
# ---------------------------------------------------------------------------

def test_module_vide():
    m = BasicModule()
    assert m.training is True
    assert list(m.children()) == []
    assert list(m.parameters()) == []


def test_parametre_enregistre_par_affectation():
    f = Feuille()
    assert f.get_param("W") is f.W
    assert f.get_param("b") is f.b


def test_enfant_enregistre_par_affectation():
    c = Conteneur()
    assert c.get_child("fc1") is c.fc1
    assert c.get_child("fc2") is c.fc2


def test_attributs_restent_accessibles():
    c = Conteneur()
    assert isinstance(c.fc1, Feuille)
    assert isinstance(c.fc1.W, BasicParameter)


def test_tenseur_ordinaire_pas_enregistre():
    """Un tensor avec requires_grad n'est pas automatiquement un parametre."""
    m = BasicModule()
    m.cache = BasicTensor(np.zeros(3), requires_grad=True)
    assert list(m.parameters()) == []


def test_autres_valeurs_pas_enregistrees():
    m = BasicModule()
    m.n = 3
    m.liste = [1, 2]
    m.nom = "test"
    assert list(m.children()) == []
    assert list(m.parameters()) == []


def test_ajout_explicite():
    m = BasicModule()
    p = BasicParameter(np.zeros(2))
    f = Feuille()
    m.add_param("p", p)
    m.add_child("f", f)
    assert m.get_param("p") is p
    assert m.get_child("f") is f


# ---------------------------------------------------------------------------
# 3. Reaffectation
# ---------------------------------------------------------------------------

def test_parametre_remplace_par_none():
    f = Feuille()
    f.b = None
    with pytest.raises(KeyError):
        f.get_param("b")
    assert len(list(f.parameters())) == 1


def test_module_remplace_par_autre_chose():
    c = Conteneur()
    c.fc2 = 42
    with pytest.raises(KeyError):
        c.get_child("fc2")
    assert len(list(c.parameters())) == 2


def test_parametre_remplace_par_module():
    m = BasicModule()
    m.x = BasicParameter(np.zeros(2))
    m.x = Feuille()
    with pytest.raises(KeyError):
        m.get_param("x")
    assert m.get_child("x") is m.x


def test_parametre_remplace_par_un_autre():
    f = Feuille()
    nouveau = BasicParameter(np.ones((2, 3)))
    f.W = nouveau
    assert f.get_param("W") is nouveau
    assert len(list(f.parameters())) == 2


# ---------------------------------------------------------------------------
# 4. Oubli de super().__init__()
# ---------------------------------------------------------------------------

def test_module_avant_super_init_leve():
    class Mauvais(BasicModule):
        def __init__(self):
            self.fc = Feuille()
            super().__init__()

    with pytest.raises((AttributeError, ValueError)):
        Mauvais()


def test_parametre_avant_super_init_leve():
    class Mauvais(BasicModule):
        def __init__(self):
            self.W = BasicParameter(np.zeros(2))
            super().__init__()

    with pytest.raises((AttributeError, ValueError)):
        Mauvais()


def test_attribut_ordinaire_avant_super_init_autorise():
    class Ok(BasicModule):
        def __init__(self):
            self.n = 3
            super().__init__()

    assert Ok().n == 3


# ---------------------------------------------------------------------------
# 5. Acces par nom
# ---------------------------------------------------------------------------

def test_get_child_inexistant():
    with pytest.raises(KeyError):
        Conteneur().get_child("fc3")


def test_get_param_inexistant():
    with pytest.raises(KeyError):
        Feuille().get_param("gamma")


def test_erreur_sans_exception_enchainee():
    """une seule exception, pas de 'During handling of the above...'."""
    with pytest.raises(KeyError) as exc:
        Conteneur().get_child("fc3")
    assert exc.value.__suppress_context__
    assert exc.value.__cause__ is None


def test_get_submodule_un_niveau():
    c = Conteneur()
    assert c.get_submodule("fc1") is c.fc1


def test_get_submodule_imbrique():
    p = Profond()
    assert p.get_submodule("bloc") is p.bloc
    assert p.get_submodule("bloc.fc2") is p.bloc.fc2


def test_get_submodule_echec_indique_le_chemin():
    with pytest.raises(KeyError) as exc:
        Profond().get_submodule("bloc.fc9")
    assert "bloc.fc9" in str(exc.value)


def test_get_subparam_imbrique():
    p = Profond()
    assert p.get_subparam("bloc.fc1.W") is p.bloc.fc1.W


def test_get_subparam_au_premier_niveau():
    p = Profond()
    assert p.get_subparam("echelle") is p.echelle


def test_get_subparam_module_inexistant():
    with pytest.raises(KeyError):
        Profond().get_subparam("bloc.fc9.W")


def test_get_subparam_parametre_inexistant():
    with pytest.raises(KeyError):
        Profond().get_subparam("bloc.fc1.gamma")
    with pytest.raises(KeyError):
        Profond().get_subparam("gamma")


# ---------------------------------------------------------------------------
# 6. children
# ---------------------------------------------------------------------------

def test_children_directs():
    p = Profond()
    assert list(p.children()) == [p.bloc]


def test_children_recursif_inclut_toutes_les_feuilles():
    """enfants sans enfants sont renvoyes."""
    c = Conteneur()
    assert list(c.children(recursive=True)) == [c.fc1, c.fc2]


def test_children_recursif_profond_en_preordre():
    p = Profond()
    assert list(p.children(recursive=True)) == [p.bloc, p.bloc.fc1, p.bloc.fc2]


def test_children_recursif_sans_doublon():
    m = BasicModule()
    f = Feuille()
    m.a = f
    m.b = f
    assert list(m.children(recursive=True)) == [f]


def test_children_recursif_appels_independants():
    p = Profond()
    premier = list(p.children(recursive=True))
    second = list(p.children(recursive=True))
    assert premier == second
    assert len(second) == 3


def test_children_ne_renvoie_pas_self():
    p = Profond()
    assert p not in list(p.children(recursive=True))


# ---------------------------------------------------------------------------
# 7. parameters
# ---------------------------------------------------------------------------

def test_parameters_feuille():
    f = Feuille()
    params = list(f.parameters())
    assert len(params) == 2
    assert params[0] is f.W and params[1] is f.b


def test_parameters_mlp_quatre_parametres():
    assert len(list(Conteneur().parameters())) == 4


def test_parameters_recursif_par_defaut():
    assert len(list(Conteneur().parameters())) > 0


def test_parameters_non_recursif():
    p = Profond()
    assert list(p.parameters(recursive=False)) == [p.echelle]
    assert list(Conteneur().parameters(recursive=False)) == []


def test_parameters_profond():
    p = Profond()
    params = list(p.parameters())
    attendus = [p.echelle, p.bloc.fc1.W, p.bloc.fc1.b, p.bloc.fc2.W, p.bloc.fc2.b]
    assert len(params) == 5
    for a in attendus:
        assert any(a is q for q in params)


def test_parameters_deux_modules_meme_nom_de_parametre():
    """check qu'il n'y pas de bug de deduplication par nom : fc1.W et fc2.W s'appellent tous les deux 'W'."""
    c = Conteneur()
    params = list(c.parameters())
    assert any(c.fc1.W is q for q in params)
    assert any(c.fc2.W is q for q in params)


def test_parameters_poids_partage_une_seule_fois():
    """Weight tying : le meme Parameter a deux endroits."""
    c = Conteneur()
    partage = BasicParameter(np.zeros((2, 3)))
    c.fc1.W = partage
    c.fc2.W = partage
    params = list(c.parameters())
    assert sum(1 for q in params if q is partage) == 1
    assert len(params) == 3


def test_parameters_module_partage_une_seule_fois():
    m = BasicModule()
    f = Feuille()
    m.a = f
    m.b = f
    assert len(list(m.parameters())) == 2


# ---------------------------------------------------------------------------
# 8. apply / train / eval
# ---------------------------------------------------------------------------

def test_apply_visite_self_et_tous_les_descendants():
    p = Profond()
    visites = []
    p.apply(visites.append)
    assert visites == [p, p.bloc, p.bloc.fc1, p.bloc.fc2]


def test_apply_renvoie_self():
    p = Profond()
    assert p.apply(lambda m: None) is p


def test_apply_module_partage_une_seule_fois():
    m = BasicModule()
    f = Feuille()
    m.a = f
    m.b = f
    visites = []
    m.apply(visites.append)
    assert visites == [m, f]


def test_eval_se_propage():
    p = Profond()
    p.eval()
    assert all(not mod.training for mod in [p, p.bloc, p.bloc.fc1, p.bloc.fc2])


def test_train_se_propage():
    p = Profond()
    p.eval()
    p.train()
    assert all(mod.training for mod in [p, p.bloc, p.bloc.fc1, p.bloc.fc2])


def test_train_avec_mode():
    p = Profond()
    p.train(False)
    assert not p.bloc.fc1.training


def test_train_eval_renvoient_self():
    p = Profond()
    assert p.eval() is p
    assert p.train() is p


# ---------------------------------------------------------------------------
# 9. __call__ / forward
# ---------------------------------------------------------------------------

def test_forward_non_defini():
    with pytest.raises(NotImplementedError):
        SansForward()(BasicTensor([1.0]))


def test_call_renvoie_le_resultat_du_forward():
    f = Feuille(2, 3)
    x = BasicTensor(np.random.randn(4, 2))
    assert np.allclose(f(x).data, f.forward(x).data)
    assert f(x).data.shape == (4, 3)


def test_call_transmet_args_et_kwargs():
    m = AvecArguments()
    assert m(1) == (1, 0, 0)
    assert m(1, 2) == (1, 2, 0)
    assert m(1, b=2, c=3) == (1, 2, 3)


# ---------------------------------------------------------------------------
# 10. Integration avec l'autograd
# ---------------------------------------------------------------------------

def test_backward_remplit_tous_les_parametres():
    p = Profond()
    x = BasicTensor(np.random.randn(5, 2))
    p(x).sum().backward()
    for param in p.parameters():
        assert param.grad is not None
        assert param.grad.shape == param.data.shape


def test_zero_grad_vide_tous_les_gradients():
    p = Profond()
    p(BasicTensor(np.random.randn(5, 2))).sum().backward()
    p.zero_grad()
    assert all(param.grad is None for param in p.parameters())


def test_zero_grad_renvoie_self():
    p = Profond()
    assert p.zero_grad() is p


def test_gradients_s_accumulent_sans_zero_grad():
    f = Feuille()
    x = BasicTensor(np.random.randn(3, 2))
    f(x).sum().backward()
    premier = f.W.grad.copy()
    f(x).sum().backward()
    assert np.allclose(f.W.grad, 2 * premier, atol=1e-5)


def test_un_pas_de_descente_fait_baisser_la_loss():
    np.random.seed(0)
    c = Conteneur()
    x = BasicTensor(np.random.randn(16, 2))
    cible = BasicTensor(np.random.randn(16, 1))

    def loss():
        return ((c(x) - cible) ** 2).mean()

    avant = loss()
    avant.backward()
    for param in c.parameters():
        param.data -= 0.01 * param.grad
    c.zero_grad()

    assert loss().data < avant.data