"""
Interface de prédictions de chiffres avec le MLP MNIST en temps réel.

Lancement   : uv run python -m src.s03_mnist.dessin

Met environ 30s a preparer le modèle.
"""
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from src.s00_autograd_and_tensors.autograd import BasicTensor, no_grad
from src.s03_mnist.mlp_mnist import get_trained_mnist

CENTRER = True  # recentre le chiffre comme dans MNIST


class DrawingPad:
    def __init__(self, root, size=280, brush=20, out_size=28, on_update=None, delay_ms=30):
        """
        zone de dessin pour rentrer les chiffres manuscrits

        :param root:
        :param size:
            taille de la zone de dessin en pixels
        :param brush:
            taille du pinceau
        :param out_size:
            taille de l'image de sortie
        :param on_update:
            fonction a appeler a chaque modif
        :param delay_ms:
            delais minimum entre deux update
        """
        self.size = size
        self.brush = brush
        self.out_size = out_size
        self.on_update = on_update
        self.delay_ms = delay_ms

        # Zone de dessin
        self.canvas = tk.Canvas(root, width=size, height=size, bg="black", cursor="cross")
        self.canvas.pack(padx=10, pady=10)

        # Image pour le modele
        self.image = Image.new("L", (size, size), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.last = None
        self._pending = False

        # mapping des touches pour le dessin
        self.canvas.bind("<Button-1>", self._start)
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._stop)

        tk.Button(root, text="Effacer", command=self.clear).pack(fill="x", padx=10, pady=(0, 10))

    # --- Dessin -----------------------------------------------------------
    def _dot(self, x, y):
        """
        le point du pinceau
        :param x:
        :param y:
        :return:
        """
        r = self.brush / 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _start(self, e):
        """
        debut de dessin
        :param e:
        :return:
        """
        self.last = (e.x, e.y)
        self._dot(e.x, e.y)
        self._schedule_update()

    def _paint(self, e):
        """
        continue de dessiner pendant qu'on bouge la souris
        :param e:
        :return:
        """
        if self.last is None:
            return
        x, y = e.x, e.y

        # ligne entre l'ancienne pos et la nouvelle
        self.canvas.create_line(*self.last, x, y, fill="white", width=self.brush,
                                capstyle=tk.ROUND, joinstyle=tk.ROUND)
        self.draw.line([self.last, (x, y)], fill=255, width=self.brush)

        self._dot(x, y)  # arrondit la "fin" du trait
        self.last = (x, y)
        self._schedule_update()

    def _stop(self, _e):
        self.last = None

    def clear(self):
        self.canvas.delete("all")

        # rectangle noir car mnist c'est chiffre blanc sur fond noir
        self.draw.rectangle([0, 0, self.size, self.size], fill=0)
        self._schedule_update()

    # --- Récupération de l'image -----------------------------------------
    def get_image(self, normalize=True):
        """
        Renvoie l'image pour le MLP de taille out_size x out_size en tant que array numpy

        :param normalize:
            si on doit normaliser dans [0, 1] ou pas
        :return:
            l'image pour le MLP
        """
        small = self.image.resize((self.out_size, self.out_size), Image.LANCZOS)
        arr = np.asarray(small, dtype=np.float32)

        return arr / 255.0 if normalize else arr

    def get_full_image(self):
        """
        Renvoie l'image taille complète
        :return:
            L'image
        """
        return np.asarray(self.image).copy()

    def _schedule_update(self):
        """
        Fait la maj après le delais et desactive la maj en attendant
        """
        if self.on_update and not self._pending:
            self._pending = True
            self.canvas.after(self.delay_ms, self._fire_update)

    def _fire_update(self):
        """
        Fait la maj et reactive les maj
        """
        self._pending = False
        self.on_update(self.get_image())


# --- traitement de l'image pour que ca ressemlbe aux images MNIST ------------------
def preprocess_mnist(full):
    """
    Prend l'image taille complète et la modifie pour qu'elle resemble aux images de MNIST
    - On réduit dans une boite de 20x20 et on centre sur le centre du dessin
    - On ajoute une bordure pour que ca fasse 28 x 28

    :param full:
        l'image
    :return:
    """
    # pixels x, y qui sont pas noir
    ys, xs = np.nonzero(full > 0)
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.float32)

    # on resize pour avoir le plus petit rectangle qui contient tous les pixels pas noir
    crop = full[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = crop.shape

    # resize en 20x20
    s = 20 / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    a = np.asarray(Image.fromarray(crop).resize((nw, nh), Image.LANCZOS), dtype=np.float32)

    # "Centre de masse" placé au centre (14, 14) de l'image 28x28
    total = a.sum() or 1.0
    cy = (a.sum(axis=1) * np.arange(nh)).sum() / total # centre des 'x'
    cx = (a.sum(axis=0) * np.arange(nw)).sum() / total # centre des 'y'

    top = int(np.clip(round(14 - cy), 0, 28 - nh))
    left = int(np.clip(round(14 - cx), 0, 28 - nw))

    out = np.zeros((28, 28), dtype=np.float32)
    out[top:top + nh, left:left + nw] = a

    return out / 255.0


# --- Appel du modèle -----------------------------------------------------
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def get_probs(model, img):
    """
    Renvoie les 10 probabilités du modèle pour une image 28x28 dans [0, 1].

    Le modèle attend des pixels bruts, dans [0, 255] comme MNIST : sa couche Normalize
    fait le reste. Il attend aussi un axe de batch : (1, 28, 28). Flatten le transforme
    en (1, 784). Il renvoie des logits, qu'on passe dans une softmax.
    """
    # passage en basic tensor pour le modèle
    x = BasicTensor((img * 255.0).reshape(1, 28, 28))
    with no_grad():
        logits = model(x).data.ravel()

    if logits.size != 10:
        raise ValueError(f"Le modèle renvoie {logits.size} valeurs au lieu de 10")

    # proba des trucs
    return softmax(logits.astype(np.float64))


# --- Panneau des probabilités --------------------------------------------
class ProbBars:
    def __init__(self, root, n=10, width=200, row_h=20):
        self.width, self.row_h, self.label_w = width, row_h, 30
        self.pred_label = tk.Label(root, text="...", font=("Arial", 48, "bold"))
        self.pred_label.pack()
        self.conf_label = tk.Label(root, text="Dessiner un chiffre", font=("Arial", 12))
        self.conf_label.pack(pady=(0, 8))

        self.canvas = tk.Canvas(root, width=width, height=n * row_h, highlightthickness=0)
        self.canvas.pack(padx=10)
        self.bars, self.texts = [], []

        # bar de pourcentage
        for i in range(n):
            y0, y1 = i * row_h + 3, (i + 1) * row_h - 3
            self.canvas.create_text(self.label_w / 2, (y0 + y1) / 2, text=str(i), font=("Arial", 12, "bold"))
            self.canvas.create_rectangle(self.label_w, y0, width - 50, y1, fill="#eeeeee", outline="")
            self.bars.append(self.canvas.create_rectangle(self.label_w, y0, self.label_w, y1, fill="#4a90d9", outline=""))
            self.texts.append(self.canvas.create_text(width - 5, (y0 + y1) / 2, text="", anchor="e", font=("Arial", 10)))

    def update(self, probs):
        # affichage meilleur guess et confiance
        if probs is None:  # dessin vide
            self.pred_label.config(text="...")
            self.conf_label.config(text="Dessiner un chiffre")
            probs = np.zeros(len(self.bars))
            best = -1
        else:
            best = int(np.argmax(probs))
            self.pred_label.config(text=str(best))
            self.conf_label.config(text=f"confiance : {probs[best]:.1%}")

        # remplissage des barres
        max_w = self.width - 50 - self.label_w
        for i, p in enumerate(probs):
            y0, y1 = i * self.row_h + 3, (i + 1) * self.row_h - 3
            self.canvas.coords(self.bars[i], self.label_w, y0, self.label_w + p * max_w, y1)
            self.canvas.itemconfig(self.bars[i], fill="#e8590c" if i == best else "#4a90d9")
            self.canvas.itemconfig(self.texts[i], text=f"{p:.1%}" if best >= 0 else "")


# --- Application ---------------------------------------------------------
if __name__ == "__main__":
    # chargement modele
    model = get_trained_mnist()

    root = tk.Tk()
    root.title("Reconnaissance de chiffres")

    left = tk.Frame(root)
    left.pack(side="left", anchor="n")
    right = tk.Frame(root)
    right.pack(side="left", anchor="n", padx=10, pady=10)

    bars = ProbBars(right)
    tk.Label(right, text="Vue du modèle").pack(pady=(12, 0))
    preview = tk.Label(right)
    preview.pack()

    def on_update(img):
        x = preprocess_mnist(pad.get_full_image()) if CENTRER else img

        if x.max() == 0:
            bars.update(None)
        else:
            bars.update(get_probs(model, x))

        zoom = Image.fromarray((x * 255).astype(np.uint8)).resize((84, 84), Image.NEAREST)
        preview.photo = ImageTk.PhotoImage(zoom)  # garde la ref sinon l'image disparait
        preview.config(image=preview.photo)

    pad = DrawingPad(left, on_update=on_update)
    on_update(pad.get_image())  # etat de base

    root.mainloop()