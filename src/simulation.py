from board import GameOfLifeBase
from croissance import Fonction_de_croissance
from filtre import Filtre
from constantes import *
from type_croissance import Type_de_croissance

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp


class Simulation:
    def __init__(self, size: int) -> None:
        self.size = size
        self.game = GameOfLifeBase(size=self.size, void=VOID_BOARD)
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

    def seed_blob(self, center: tuple, radius: float, amplitude: float = 1.0) -> None:
        """Ajouter un blob gaussien au centre `center` (y, x) avec `radius` en pixels."""
        y, x = np.ogrid[0 : self.size, 0 : self.size]
        cy, cx = center
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        blob = amplitude * np.exp(-(r**2) / (2 * (radius**2)))
        self.X += blob
        self.X = np.clip(self.X, 0, 1)

    def seed_ring(
        self, center: tuple, radius: float, width: float = 2.0, amplitude: float = 1.0
    ) -> None:
        """Ajouter un anneau (orbion-like) au centre `center` avec `radius` et `width`."""
        y, x = np.ogrid[0 : self.size, 0 : self.size]
        cy, cx = center
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        ring = amplitude * np.exp(-((r - radius) ** 2) / (2 * (width**2)))
        self.X += ring
        self.X = np.clip(self.X, 0, 1)

    def seed_random_orbions(
        self, n: int = 3, radius_range=(5, 20), width_range=(1.5, 4.0)
    ) -> None:
        """Placer `n` anneaux aléatoires (orbions) sur la grille."""
        for _ in range(n):
            cy = np.random.randint(0, self.size)
            cx = np.random.randint(0, self.size)
            radius = np.random.uniform(*radius_range)
            width = np.random.uniform(*width_range)
            amp = np.random.uniform(0.6, 1.0)
            self.seed_ring((cy, cx), radius, width, amp)

    def seed_glider(
        self, center: tuple, orientation: int = 0, amplitude: float = 1.0
    ) -> None:
        """Placer un planeur classique (Game of Life) centré sur `center`.

        - `center`: (y, x) coordonnées du centre du planeur
        - `orientation`: 0..3 rotations de 90° (sens horaire)
        - `amplitude`: valeur (0..1) assignée aux cellules vivantes
        """
        # motif de base (glider classique 3x3)
        pattern = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
        # appliquer rotation
        pattern = np.rot90(pattern, -orientation)

        h, w = pattern.shape
        cy, cx = center
        # top-left pour coller le motif centré
        top = int(np.round(cy - h // 2))
        left = int(np.round(cx - w // 2))

        for dy in range(h):
            for dx in range(w):
                y = (top + dy) % self.size
                x = (left + dx) % self.size
                if pattern[dy, dx] > 0:
                    self.X[y, x] = np.clip(self.X[y, x] + amplitude, 0, 1)

    def seed_orbium(
        self,
        orbium=ORBIUM,
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
