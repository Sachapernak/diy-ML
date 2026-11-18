import numpy as np

from src.s00_autograd_and_tensors.autograd import BasicTensor
from src.s01_module_and_init.layers import Sequential, Linear, ReLU
from src.s02_optimizer.optimizer import SGD

# XOR, en NumPy
x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float32)

# Valeur du XOR pour chaque point (vrai si les deux entrées sont différentes)
y = (x[:, 0] != x[:, 1]).astype(np.float32)          # [0, 1, 1, 0]

ENTREES = BasicTensor(x)                              # (4, 2)

# MSE avec une sortie par point -> shape (4, 1)
CIBLES_MSE = BasicTensor(y.reshape(4, 1))             # [[0], [1], [1], [0]]

# CE avec deux classes -> indice de la classe == valeur du XOR
#   faux (0) -> [1, 0]
#   vrai (1) -> [0, 1]
CIBLES_ONE_HOT = BasicTensor(np.eye(2, dtype=np.float32)[y.astype(int)])   # (4, 2)

N_STEP = 1000

LR = 0.01

if __name__ == '__main__':

    mlp = Sequential(
        Linear(2, 8),
        ReLU(),
        Linear(8, 1),
    )

    mlp_one_hot = Sequential(
        Linear(2, 8),
        ReLU(),
        Linear(8, 2),
    )

    sgd_mlp1 = SGD(mlp.parameters(), learning_rate=LR)
    sgd_mlp2 = SGD(mlp_one_hot.parameters(), learning_rate=LR)



    for step in range(N_STEP):
        logits_mlp1 = mlp(ENTREES)
        logits_mlp2 = mlp_one_hot(ENTREES)

        loss_mlp1 = logits_mlp1.mse_loss(CIBLES_MSE)
        loss_mlp2 = logits_mlp2.cross_entropy(CIBLES_ONE_HOT)

        sgd_mlp1.zero_grad()
        sgd_mlp2.zero_grad()

        loss_mlp1.backward()
        loss_mlp2.backward()

        sgd_mlp1.step()
        sgd_mlp2.step()

        if step % (N_STEP // 12) == 0 and (step != N_STEP - 1):
            print(f"Étape {step:>5} | loss mlp1 (MSE) : {float(loss_mlp1.data):.3e} | "
                  f"loss mlp2 (CE) : {float(loss_mlp2.data):.3e}")

        if step == N_STEP - 1:
            print(f"Étape {(N_STEP - 1):>5} | loss mlp1 (MSE) : {float(loss_mlp1.data):.3e} | "
                  f"loss mlp2 (CE) : {float(loss_mlp2.data):.3e}")

    prediction_mlp1 = mlp(ENTREES).data.ravel()
    prediction_mlp2 = mlp_one_hot(ENTREES).data.argmax(axis=1)
    print()
    print(f"{'attendu':<16} : {y.astype(int)}")
    print(f"{'mlp1 (MSE)':<16} : {(prediction_mlp1 > 0.5).astype(int)}")
    print(f"{'mlp2 (CE)':<16} : {prediction_mlp2}")



