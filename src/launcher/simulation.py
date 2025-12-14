from board.board import Board
from croissance.croissances import Fonction_de_croissance
from kernel.filtre import Filtre
from const.constantes import *
from croissance.type_croissance import Type_de_croissance

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import shift
from species.species_types import Species_types


class Simulation:
    def __init__(
        self, size: int = BOARD_SIZE, kernel_type: Species_types = Species_types.GENERIC
    ) -> None:
        """
        Initialiser la simulation.
        kernel_type: "orbium", "hydrogeminium", ou "generic"
        """
        self.size = size
        self.game = Board()

        # Choisir le kernel selon le type
        if kernel_type == Species_types.HYDROGEMINIUM:
            b = HYDROGEMINIUM_B
            self.filtre = Filtre(
                fonction_de_croissance=Fonction_de_croissance(
                    type=Type_de_croissance.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=b,  # Utiliser le kernel multi-anneau
            )
        elif kernel_type == Species_types.ORBIUM:
            b = ORBIUM_B
            self.filtre = Filtre(
                fonction_de_croissance=Fonction_de_croissance(
                    type=Type_de_croissance.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=b,
            )
        elif kernel_type == Species_types.FISH:
            self.filtre = Filtre(
                fonction_de_croissance=Fonction_de_croissance(
                    type=Type_de_croissance.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=None,
                kernels=FISH_KERNEL
            )
            self.filtre.kernels = FISH_KERNEL  # Utiliser le kernel défini pour fish 
        else:  # generic
            self.filtre = Filtre(
                fonction_de_croissance=Fonction_de_croissance(
                    type=Type_de_croissance.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                mus=MUS,
                sigmas=SIGMAS,
            )

        # board initial (copie pour pouvoir modifier sans toucher à l'objet GameOfLifeBase)
        self.X = self.game.get_board.copy()

    def apply_patch(
        self,
        patch: np.ndarray,
        center: tuple | None = None,
        rotate: int = 0,
        amplitude: float = 1.0,
        normalize: bool = True,
    ) -> None:
        """
        Applique un patch 2D sur la grille self.X.

        patch     : np.ndarray, valeurs approximativement 0..1
        center    : (y,x) coordonnée du centre où placer le patch, par défaut centre de la grille
        rotate    : nombre de rotations de 90° (0..3)
        amplitude : facteur multiplicatif appliqué au patch
        normalize : si True, normalise le patch à 1 avant amplitude
        """

        arr = np.array(patch, dtype=float)
        if rotate % 4 != 0:
            arr = np.rot90(arr, -(rotate % 4))

        if normalize:
            mx = arr.max()
            if mx > 0:
                arr /= mx

        #  recentrage
        yy, xx = np.indices(arr.shape)
        mass = arr.sum()

        if mass > 0:
            cy = (yy * arr).sum() / mass
            cx = (xx * arr).sum() / mass
        else:
            cy = (arr.shape[0] - 1) / 2
            cx = (arr.shape[1] - 1) / 2

        gy = (arr.shape[0] - 1) / 2
        gx = (arr.shape[1] - 1) / 2
        dy = gy - cy
        dx = gx - cx

        arr = shift(
            arr, shift=(dy, dx), order=1, mode="constant", cval=0.0, prefilter=False
        )

        # appliquer amplitude
        arr *= amplitude

        # calcul du coin supérieur gauche
        h, w = arr.shape
        if center is None:
            cy, cx = self.size // 2, self.size // 2
        else:
            cy, cx = center

        top = int(round(cy - h / 2))
        left = int(round(cx - w / 2))

        # injection du patch dans la grille avec wrap
        for dy in range(h):
            for dx in range(w):
                y = (top + dy) % self.size
                x = (left + dx) % self.size
                self.X[y, x] = np.clip(self.X[y, x] + arr[dy, dx], 0, 1)

    def __update(self, frame: int) -> list:
        self.X = self.filtre.evolve_lenia(self.X)
        self.img.set_data(self.X)
        return [self.img]

    def run(self):
        fig, ax = plt.subplots()
        self.img = ax.imshow(self.X, cmap="inferno", interpolation="none")
        ax.set_title("Lenia")
        ax.set_xticks([])
        ax.set_yticks([])
        anim = animation.FuncAnimation(
            fig, self.__update, frames=200, interval=20, blit=True
        )
        plt.show()
