import numpy as np

from src.s00_autograd_and_tensors.autograd import BasicTensor, no_grad
from src.s01_module_and_init.layers import Sequential, Linear, ReLU
from src.s02_optimizer.optimizer import SGD, MomentumGradient, Adam, AdamW

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
MAX_PAS = 1000
AFFICHAGE_TOUS_LES = 200

NB_SEED = 25


# ---------------------------------------------------------------------------
# Optimiseurs à comparer
# ---------------------------------------------------------------------------

# Un optimiseur se construit à partir des paramètres d'un modèle précis :
# on stocke donc une fabrique (params -> optimiseur), pas un optimiseur déjà créé.
OPTIMISEURS = {
    "SGD":          lambda params: SGD(params, learning_rate=LR),
    "Momentum 0.9": lambda params: MomentumGradient(params, learning_rate=LR, momentum=0.9),
    # Avec momentum 0.9, pas effectif vaut environ lr / (1-0.9) = 10*lr.
    # variante pour comparer avec pas equivalent
    "Momentum 0.9 (lr/10)": lambda params: MomentumGradient(params, learning_rate=LR / 10, momentum=0.9),
    # Adam : le pas vaut environ lr par paramètre, quelle que soit la taille du gradient.
    "Adam (lr 1e-3)": lambda params: Adam(params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8),
    # lr = 1e-3 est la valeur par défaut, qui marche bien pour les gros modèles, mais sur un petit XOR,
    # il faut beaucoup de pas. Ici on test aussi avec 1e-2 pour que ce soit plus raisonnable
    "Adam (lr 1e-2)": lambda params: Adam(params, learning_rate=1e-2, beta1=0.9, beta2=0.999, eps=1e-8),
    "AdamW (lr 1e-2)": lambda params: AdamW(params, learning_rate=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, decay=1e-2),
}


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


VARIANTES = [
    ("MSE", mlp_mse, loss_mse, prediction_mse),
    ("CE",  mlp_ce,  loss_ce,  prediction_ce),
]

# permet de voir les neurones cachés
def inspecter(model):
    with no_grad():
        pre = model[0](ENTREES).data  # avant ReLU
        actifs = pre > 0  # vrai si le neurone laisse passer

    for point, ligne in zip(x.astype(int), actifs.astype(int)):
        print(f"    point {point} : {ligne}  ({ligne.sum()} actifs)")

    morts = ~actifs.any(axis=0)  # aucun actif
    print(f"    neurones morts partout : {np.flatnonzero(morts)}")


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
            return max_pas

    return None


def lancer(nom, fabrique_modele, calcul_loss, prediction, nom_optim, fabrique_optim,
           seed, verbose=False, afficher=True):
    """
    Entraîne un modèle avec un optimiseur, pour une seed

    :param fabrique_optim: params -> optimiseur
    :return: nb de pas necessaires ou None en cas d'échec
    """
    rng = np.random.default_rng(seed)
    model = fabrique_modele(rng)
    optim = fabrique_optim(model.parameters())

    if verbose:
        print(f"Loss pour : {nom} / {nom_optim}")

    pas = entrainer_jusqu_a_reussite(model, optim, calcul_loss, prediction, verbose=verbose)

    if afficher:
        with no_grad():
            predites = prediction(model(ENTREES))
            loss_finale = float(calcul_loss(model(ENTREES)).data)

        resultat = f"{pas} pas" if pas is not None else f"échec (>{MAX_PAS})"
        print(f"  {nom:<4} | {nom_optim:<20} | seed {seed:>2} | {resultat:<17} "
              f"| loss : {loss_finale:.3e} | prédit : {predites}")
    return pas


def rejouer_et_inspecter(seed, titre):
    """Rejoue une seed en CE avec SGD, et inspecte les neurones cachés à quelques étapes."""
    rng = np.random.default_rng(seed)
    model = mlp_ce(rng)
    optim = SGD(model.parameters(), learning_rate=LR)
    print(f"\n == {titre} (seed {seed}) ==")

    for pas in range(501):
        if pas in (1, 500):
            with no_grad():
                loss = float(loss_ce(model(ENTREES)).data)
            print(f"étape {pas} | loss {loss:.4f}")
            inspecter(model)

        loss = loss_ce(model(ENTREES))
        optim.zero_grad()
        loss.backward()
        optim.step()


if __name__ == '__main__':


    print("\n====================================")
    print(f"== Test sur une seule seed:")
    print("====================================")

    print(f"Nombre de pas max : {MAX_PAS}")

    print(f"Résultat Attendu : {Y_INT}\n")

    # Un run pour chaque variante et chaque optimiseur
    for nom, fabrique, calcul_loss, prediction in VARIANTES:
        for nom_optim, fabrique_optim in OPTIMISEURS.items():
            lancer(nom, fabrique, calcul_loss, prediction, nom_optim, fabrique_optim,
                   seed=0, verbose=True)
            print()


    # Mêmes seed pour tous les optimiseurs -> mêmes init
    print("\n======================================")
    print(f"== Test sur {NB_SEED} seed :")
    print("======================================")
    resume = []
    for nom, fabrique, calcul_loss, prediction in VARIANTES:
        for nom_optim, fabrique_optim in OPTIMISEURS.items():
            resultats = [lancer(nom, fabrique, calcul_loss, prediction, nom_optim, fabrique_optim,
                                seed, afficher=False)
                         for seed in range(NB_SEED)]
            reussis = [p for p in resultats if p is not None]
            mediane = np.median(reussis) if reussis else float("nan")
            resume.append((nom, nom_optim, len(reussis), mediane))

    print(f"  {'loss':<4} | {'optimiseur':<20} | réussis | médiane")
    for nom, nom_optim, nb_reussis, mediane in resume:
        print(f"  {nom:<4} | {nom_optim:<20} | {nb_reussis:>3}/{NB_SEED:<3} | {mediane:.0f} pas")

    print("\n======================================")
    print(f"== Test sur seed specifiques :")
    print("======================================")

    rejouer_et_inspecter(2, "Verification d'une seed qui fonctionne pas")
    # On observe :
    # - tous les neurons ont les meme activation (soit s'activent tout le temps, soit jamais)
    # - les relus qui restent active tout le temps introduisent pas de non linearité
    # - le modele devient linéaire. Un xor peut pas être representé par un modèle linéaire
    # - donc ca marche pas

    rejouer_et_inspecter(1, "Verification d'une seed qui fonctionne")
    # On observe :
    # - Le neuron '0' s'active que quand on a les deux entrée à 1 : c'est un AND
    # - les neurons 1, 6 ne servent a rien car il sont mort
    # - le neuron 4 permet d'ajouter un terme lineaire dans la sortie (pas forcément utile ici)
    # - le neuron 3 s'active que pour [0, 1], peut etre utile
    # - le neuron 2, 5 : sont des OR
    # - le neuron 7 replique la deuxime entrée
    # -> ce sont probablement 0, 2, 3, 5 qui sont réellement utile ici