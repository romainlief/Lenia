from board import GameOfLifeBase
from croissance import Fonction_de_croissance
from filtre import Filtre
from constantes import *
from type_croissance import Type_de_croissance

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp
from species import Species


class Simulation:
    def __init__(self, size: int, kernel_type: Species = Species.GENERIC) -> None:
        """
        Initialiser la simulation.
        kernel_type: "orbium", "hydrogeminium", ou "generic"
        """
        self.size = size
        self.game = GameOfLifeBase(size=self.size, void=VOID_BOARD)

        # Choisir le kernel selon le type
        if kernel_type == Species.HYDROGEMINIUM:
            b = pattern["geminium"].get("b")
            self.filtre = Filtre(
                fonction_de_croissance=Fonction_de_croissance(
                    type=Type_de_croissance.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=b,  # Utiliser le kernel multi-anneau
            )
        elif kernel_type == Species.ORBIUM:
            b = pattern["orbium"].get("b")
            print("lol")
            self.filtre = Filtre(
                fonction_de_croissance=Fonction_de_croissance(
                    type=Type_de_croissance.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=b,
            )
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

    def seed_orbium(
        self,
        orbium=pattern["orbium"].get("cells"),
        top_left: tuple | None = None,
        center: tuple | None = None,
        rotate: int = 0,
        amplitude: float = 1.0,
        normalize: bool = True,
    ) -> None:
        """Insérer un motif `orbium` (numpy 2D) dans la grille `self.X`.

        - `orbium`: tableau 2D (valeurs ≈ 0..1).
        - `top_left`: (y, x) coordonnée du coin supérieur gauche où coller le motif.
        - `center`: (y, x) coordonnée du centre du motif (utilisé si `top_left` est None).
        - `rotate`: nombre de rotations de 90° (0..3, sens horaire).
        - `amplitude`: facteur multiplicatif appliqué au motif.
        - `normalize`: si True, normalise l'orbium pour que sa valeur max soit 1 avant `amplitude`.

        Si `top_left` et `center` sont None, le motif est collé au centre de la grille.
        Les bords sont enroulés (modulo) pour éviter les erreurs d'indice.
        """
        arr = np.array(orbium, dtype=float)
        if rotate % 4 != 0:
            arr = np.rot90(arr, -(rotate % 4))

        if normalize:
            mx = arr.max()
            if mx > 0:
                arr = arr / mx

        arr = arr * amplitude

        h, w = arr.shape

        if top_left is None:
            if center is None:
                cy = self.size // 2
                cx = self.size // 2
            else:
                cy, cx = center
            top = int(np.round(cy - h // 2))
            left = int(np.round(cx - w // 2))
        else:
            top, left = top_left

        for dy in range(h):
            for dx in range(w):
                y = (top + dy) % self.size
                x = (left + dx) % self.size
                self.X[y, x] = np.clip(self.X[y, x] + arr[dy, dx], 0, 1)

    def seed_hydrogeminium(
        self,
        hydrogeminium=pattern["geminium"].get("cells"),
        top_left: tuple | None = None,
        center: tuple | None = None,
        rotate: int = 0,
        amplitude: float = 1.0,
        normalize: bool = True,
    ) -> None:
        """Insérer un motif `hydrogeminium` (numpy 2D) dans la grille `self.X`.

        - `hydrogeminium`: tableau 2D (valeurs ≈ 0..1).
        - `top_left`: (y, x) coordonnée du coin supérieur gauche où coller le motif.
        - `center`: (y, x) coordonnée du centre du motif (utilisé si `top_left` est None).
        - `rotate`: nombre de rotations de 90° (0..3, sens horaire).
        - `amplitude`: facteur multiplicatif appliqué au motif.
        - `normalize`: si True, normalise l'hydrogeminium pour que sa valeur max soit 1 avant `amplitude`.

        Si `top_left` et `center` sont None, le motif est collé au centre de la grille.
        Les bords sont enroulés (modulo) pour éviter les erreurs d'indice.
        """
        arr = np.array(hydrogeminium, dtype=float)
        if rotate % 4 != 0:
            arr = np.rot90(arr, -(rotate % 4))

        if normalize:
            mx = arr.max()
            if mx > 0:
                arr = arr / mx

        arr = arr * amplitude

        h, w = arr.shape

        if top_left is None:
            if center is None:
                cy = self.size // 2
                cx = self.size // 2
            else:
                cy, cx = center
            top = int(np.round(cy - h // 2))
            left = int(np.round(cx - w // 2))
        else:
            top, left = top_left

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
