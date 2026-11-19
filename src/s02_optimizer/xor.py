import numpy as np

from src.s00_autograd_and_tensors.autograd import BasicTensor, no_grad
from src.s01_module_and_init.layers import Sequential, Linear, ReLU
from src.s02_optimizer.optimizer import SGD

# XOR, en NumPy
x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float32)

# Valeur du XOR pour chaque point (vrai si les deux entrées sont différentes)
y = (x[:, 0] != x[:, 1]).astype(np.float32)          # [0, 1, 1, 0]
Y_INT = y.astype(int)

ENTREES = BasicTensor(x)                              # (4, 2)

# MSE avec une sortie par point -> shape (4, 1)
CIBLES_MSE = BasicTensor(y.reshape(4, 1))             # [[0], [1], [1], [0]]

# CE avec deux classes -> indice de la classe == valeur du XOR
#   faux (0) -> [1, 0]
#   vrai (1) -> [0, 1]
CIBLES_ONE_HOT = BasicTensor(np.eye(2, dtype=np.float32)[Y_INT])   # (4, 2)

LR = 0.1
MAX_PAS = 5000
AFFICHAGE_TOUS_LES = 200

NB_SEED = 25


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# MLP avec une seule sortie entre 0 et 1. Si proche de 0 : 'faux', si proche de 1 : 'vrai'
def mlp_mse(rng):
    return Sequential(
        Linear(2, 8, rng=rng),
        ReLU(),
        Linear(8, 1, rng=rng)
    )

# MLP en one_hot avec [1, 0] = faux et [0, 1] = vrai
def mlp_ce(rng):
    return Sequential(
        Linear(2, 8, rng=rng),
        ReLU(),
        Linear(8, 2, rng=rng)
    )


# loss pour le MLP avec une sortie
def loss_mse(sortie):
    return sortie.mse_loss(CIBLES_MSE)

# loss pour le MLP en one hot
def loss_ce(logits):
    return logits.cross_entropy(CIBLES_ONE_HOT)

# la sortie vaut 1 si elle vaut plus que 0,5, sinon elle vaut 0
def prediction_mse(sortie):
    return (sortie.data.ravel() > 0.5).astype(int)

# La sortie vaut 1 la ou elle est la plus grande (grace a argmax)
def prediction_ce(logits):
    return logits.data.argmax(axis=1)


# permet de voir les neurones cachés
def inspecter(model):
    with no_grad():
        pre = model[0](ENTREES).data  # avant ReLU
        actifs = pre > 0  # vrai si le neurone laisse passer

    for point, ligne in zip(x.astype(int), actifs.astype(int)):
        print(f"    point {point} : {ligne}  ({ligne.sum()} actifs)")

    morts = ~actifs.any(axis=0)  # aucun actif
    print(f"    neurones morts partout : {np.flatnonzero(morts)}")


# Rejouer une graine qui échoue, en inspectant à intervalles réguliers
rng = np.random.default_rng(2)
model = mlp_ce(rng)
optim = SGD(model.parameters(), learning_rate=LR)


# ---------------------------------------------------------------------------
# Entrainement
# ---------------------------------------------------------------------------

def entrainer_jusqu_a_reussite(model, optim, calcul_loss, prediction,
                               max_pas=MAX_PAS, verbose=False):
    """
    Entraîne le modèle jusqu'à ce que les quatre valeurs possibles soient bien classés.

    :return: nb de pas necessaires ou None si on a atteint max_pas
    """
    for pas in range(max_pas):
        # check si a on le bon resultat
        with no_grad():
            if np.array_equal(prediction(model(ENTREES)), Y_INT):
                return pas

        loss = calcul_loss(model(ENTREES))
        optim.zero_grad()
        loss.backward()
        optim.step()

        if verbose and pas % AFFICHAGE_TOUS_LES == 0:
            print(f"  Etape {pas:<6} loss : {float(loss.data):.3e}")

    with no_grad():
        if np.array_equal(prediction(model(ENTREES)), Y_INT):
            return pas

    return None


def lancer(nom, fabrique_modele, calcul_loss, prediction, seed, verbose=False):
    """
    test differents optimizer pour
    """
    rng = np.random.default_rng(seed)
    model = fabrique_modele(rng)
    optim = SGD(model.parameters(), learning_rate=LR)

    if verbose:
        print(f"Loss pour : {nom}")

    pas = entrainer_jusqu_a_reussite(model, optim, calcul_loss, prediction, verbose=verbose)

    with no_grad():
        predites = prediction(model(ENTREES))
        loss_finale = float(calcul_loss(model(ENTREES)).data)

    resultat = f"{pas} pas" if pas is not None else f"échec (>{MAX_PAS})"
    print(f"  {nom:<8} | seed {seed:>2} | {resultat:<17} | loss : {loss_finale:.3e} | prédit : {predites}")
    return pas


if __name__ == '__main__':

    print(f"Résultat Attendu : {Y_INT}\n")

    # Un run détaillé pour chaque variante
    lancer("MSE", mlp_mse, loss_mse, prediction_mse, seed=0, verbose=True)
    print()
    lancer("CE", mlp_ce, loss_ce, prediction_ce, seed=0, verbose=True)
    print()

    # Plusieurs graines : une seule initialisation peut avoir de la chance
    print(f"\nSur {NB_SEED} graines :")
    for nom, fabrique, calcul_loss, prediction in [
        ("MSE", mlp_mse, loss_mse, prediction_mse),
        ("CE", mlp_ce, loss_ce, prediction_ce),
    ]:
        resultats = [lancer(nom, fabrique, calcul_loss, prediction, graine) for graine in range(NB_SEED)]
        reussis = [p for p in resultats if p is not None]
        mediane = np.median(reussis) if reussis else float("nan")
        print(f"  -> {nom} : {len(reussis)}/{NB_SEED} réussis, médiane {mediane:.0f} pas\n")


    # Verif d'un seed qui marche pas
    rng = np.random.default_rng(2)
    model = mlp_ce(rng)
    optim = SGD(model.parameters(), learning_rate=LR)
    print("\n == Verification d'une seed qui fonctionne pas ==")

    for pas in range(2001):
        if pas in (0, 1, 5, 20, 100, 500):
            with no_grad():
                loss = float(loss_ce(model(ENTREES)).data)
            print(f"étape {pas} | loss {loss:.4f}")
            inspecter(model)

        loss = loss_ce(model(ENTREES))
        optim.zero_grad()
        loss.backward()
        optim.step()

    # On observe :
    # - tous les neurons ont les meme activation (soit s'activent tout le temps, soit jamais)
    # - les relus qui restent active tout le temps introduisent pas de non linearité
    # - le modele devient linéaire. Un xor peut pas être representé par un modèle linéaire
    # - donc ca marche pas


    # Verif d'un seed qui marche
    rng = np.random.default_rng(1)
    model = mlp_ce(rng)
    optim = SGD(model.parameters(), learning_rate=LR)

    print("\n == Verification d'une seed qui fonctionne ==")

    for pas in range(2001):
        if pas in (0, 1, 5, 20, 100, 500):
            with no_grad():
                loss = float(loss_ce(model(ENTREES)).data)
            print(f"étape {pas} | loss {loss:.4f}")
            inspecter(model)

        loss = loss_ce(model(ENTREES))
        optim.zero_grad()
        loss.backward()
        optim.step()

    # On observe :
    # - Le neuron '0' s'active que quand on a les deux entrée à 1 : c'est un AND
    # - les neurons 1, 6 ne servent a rien car il sont mort
    # - le neuron 4 permet d'ajouter un terme lineaire dans la sortie (pas forcément utile ici)
    # - le neuron 3 s'active que pour [0, 1], peut etre utile
    # - le neuron 2, 5 : sont des OR
    # - le neuron 7 replique la deuxime entrée
    # -> ce sont probablement 0, 2, 3, 5 qui sont réellement utile ici


