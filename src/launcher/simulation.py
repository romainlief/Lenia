from board.board import Board
from croissance.croissances import Fonction_de_croissance
from kernel.filtre import Filtre
from const.constantes import (
    FILTRE_SIZE,
    ORBIUM_B,
    MUS,
    SIGMAS,
    USE_AQUARIUM_PARAMS,
    AQUARIUM_CELLS,
    AQUARIUM_KERNEL,
    USE_EMITTER_PARAMS,
    EMITTER_CELLS,
    EMITTER_KERNEL,
    HYDROGEMINIUM_B,
    BOARD_SIZE,
    FISH_KERNEL,
    WANDERER_B,
    USE_PACMAN_PARAMS,
    PACMAN_CELLS,
    PACMAN_KERNEL,
    KERNEL_TYPE,
    USE_HYDROGEMINIUM_PARAMS,
    USE_ORBIUM_PARAMS,
    USE_FISH_PARAMS,
    USE_WANDERER_PARAMS,
)
from species.orbium import Orbium
from species.hydrogeminium import Hydrogeminium
from species.fish import Fish
from species.wanderer import Wanderer
from croissance.type_croissance import Type_de_croissance

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import shift
from species.species_types import Species_types


class Simulation:
    def __init__(
        self, size: int = BOARD_SIZE, kernel_type: Species_types = KERNEL_TYPE
    ) -> None:
        """
        Initialiser la simulation.
        kernel_type: "orbium", "hydrogeminium", ou "generic"
        """
        self.size = size
        self.game = Board()
        self.multi_channel: bool = self.game.channels > 1

        if USE_AQUARIUM_PARAMS:
            self.place_aquarium(
                self.game, AQUARIUM_CELLS, self.game.size // 2, self.game.size // 2
            )

        if USE_EMITTER_PARAMS:
            self.place_emitter(
                self.game, EMITTER_CELLS, self.game.size // 2, self.game.size // 2
            )

        if USE_PACMAN_PARAMS:
            self.place_pacman(
                self.game, PACMAN_CELLS, self.game.size // 4, self.game.size // 4
            )

        # Choisir le kernel selon le type
        if kernel_type == Species_types.HYDROGEMINIUM:
            b = HYDROGEMINIUM_B
            self.filtre = Filtre(
                fonction_de_croissance=Fonction_de_croissance(
                    type=Type_de_croissance.GAUSSIENNE,
                ),
                size=FILTRE_SIZE,
                b=b,  # Utiliser le kernel multi-anneau
                species_type=Species_types.HYDROGEMINIUM,
            )
        elif kernel_type == Species_types.ORBIUM:
            b = ORBIUM_B
            self.filtre = Filtre(
                fonction_de_croissance=Fonction_de_croissance(
                    type=Type_de_croissance.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=b,
                species_type=Species_types.ORBIUM,
            )
        elif kernel_type == Species_types.FISH:
            self.filtre = Filtre(
                fonction_de_croissance=Fonction_de_croissance(
                    type=Type_de_croissance.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=None,
                kernels=FISH_KERNEL,
                species_type=Species_types.FISH,
            )
            self.filtre.kernels = FISH_KERNEL  # Utiliser le kernel défini pour fish
        elif kernel_type == Species_types.AQUARIUM:
            self.filtre = Filtre(
                fonction_de_croissance=Fonction_de_croissance(
                    type=Type_de_croissance.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=None,
                kernels=AQUARIUM_KERNEL,
                multi_channel=self.multi_channel,
                species_type=Species_types.AQUARIUM,
            )
            self.filtre.kernels = (
                AQUARIUM_KERNEL  # Utiliser le kernel défini pour aquarium
            )
        elif kernel_type == Species_types.WANDERER:
            b = WANDERER_B
            self.filtre = Filtre(
                fonction_de_croissance=Fonction_de_croissance(
                    type=Type_de_croissance.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=b,
                species_type=Species_types.WANDERER,
            )
        elif kernel_type == Species_types.EMITTER:
            self.filtre = Filtre(
                fonction_de_croissance=Fonction_de_croissance(
                    type=Type_de_croissance.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=None,
                kernels=EMITTER_KERNEL,
                multi_channel=self.multi_channel,
                species_type=Species_types.EMITTER,
            )
            self.filtre.kernels = (
                EMITTER_KERNEL  # Utiliser le kernel défini pour emitter
            )
        elif kernel_type == Species_types.PACMAN:
            self.filtre = Filtre(
                fonction_de_croissance=Fonction_de_croissance(
                    type=Type_de_croissance.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=None,
                kernels=PACMAN_KERNEL,
                multi_channel=self.multi_channel,
                species_type=Species_types.PACMAN,
            )
            self.filtre.kernels = PACMAN_KERNEL
        else:  # generic
            self.filtre = Filtre(
                fonction_de_croissance=Fonction_de_croissance(
                    type=Type_de_croissance.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                mus=MUS,
                sigmas=SIGMAS,
                species_type=Species_types.GENERIC,
            )

        # board initial (copie pour pouvoir modifier sans toucher à l'objet GameOfLifeBase)
        self.X_raw = self.game.board.copy()
        # Représentation utilisée pour l'évolution : soit 2D, soit liste de plans (multi-canaux)
        if self.X_raw.ndim == 3 and self.multi_channel:
            self.X = [self.X_raw[:, :, c].copy() for c in range(self.X_raw.shape[2])]
        elif self.X_raw.ndim == 3:
            self.X = np.mean(self.X_raw, axis=2)
        else:
            self.X = self.X_raw.copy()

        if USE_ORBIUM_PARAMS:
            orbium = Orbium()
            patch = orbium.make_patch(
                rotate=0,
                amplitude=4.0,
                normalize=True,
            )
        elif USE_HYDROGEMINIUM_PARAMS:
            hydrogenium = Hydrogeminium()
            patch = hydrogenium.make_patch(
                rotate=0,
                amplitude=4.0,
                normalize=True,
            )
        elif USE_FISH_PARAMS:
            fish = Fish()
            patch = fish.make_patch(
                rotate=0,
                amplitude=4.0,
                normalize=True,
            )
        elif USE_WANDERER_PARAMS:
            wanderer = Wanderer()
            patch = wanderer.make_patch(
                rotate=0,
                amplitude=4.0,
                normalize=True,
            )
        if not USE_AQUARIUM_PARAMS and not USE_EMITTER_PARAMS and not USE_PACMAN_PARAMS:
            self.apply_patch(patch, center=(BOARD_SIZE // 2, BOARD_SIZE // 2))

    def apply_patch(
        self,
        patch: np.ndarray | None = None,
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
        if patch is None:
            return

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
                if isinstance(self.X, list):
                    for c in range(len(self.X)):
                        self.X[c][y, x] = np.clip(self.X[c][y, x] + arr[dy, dx], 0, 1)
                else:
                    self.X[y, x] = np.clip(self.X[y, x] + arr[dy, dx], 0, 1)

        # synchroniser la grille brute multi-canaux
        if hasattr(self, "X_raw") and self.X_raw.ndim == 3:
            if isinstance(self.X, list):
                self.X_raw = np.clip(np.stack(self.X, axis=2), 0, 1)
            else:
                self.X_raw = np.clip(
                    np.stack([self.X] * self.X_raw.shape[2], axis=2), 0, 1
                )

    def __update(self, frame: int) -> list:
        # évolution (gère mode multi-canaux si `self.X` est une liste)
        if isinstance(self.X, list):
            self.X = self.filtre.evolve_lenia(self.X)
            # `evolve_lenia` doit retourner une liste de plans
            stack = np.clip(np.stack(self.X, axis=2), 0, 1)
            # mettre à jour la grille brute
            if hasattr(self, "X_raw") and self.X_raw.ndim == 3:
                self.X_raw = stack.copy()
            # préparer affichage RGB si possible
            if stack.shape[2] >= 3:
                display = stack[:, :, :3]
            elif stack.shape[2] == 2:
                zeros = np.zeros_like(stack[:, :, 0])
                display = np.dstack((stack[:, :, 0], stack[:, :, 1], zeros))
            else:
                display = stack[:, :, 0]
        else:
            self.X = self.filtre.evolve_lenia(self.X)
            display = self.X
        self.img.set_data(display)
        return [self.img]

    def run(self, num_steps=100, interpolation="bicubic"):
        if self.multi_channel:
            self.__run_multi(num_steps=num_steps, interpolation=interpolation)
        else:
            self.__run()

    def __run(self):
        fig, ax = plt.subplots()
        self.img = ax.imshow(self.X, cmap="inferno", interpolation="none")
        ax.set_title("Lenia")
        ax.set_xticks([])
        ax.set_yticks([])
        anim = animation.FuncAnimation(
            fig, self.__update, frames=200, interval=20, blit=True
        )
        plt.show()

    def __run_multi(self, num_steps=100, interpolation="bicubic"):
        if not isinstance(self.X, list):
            raise RuntimeError(
                "run_multi requires multi-channel board (self.X as list)"
            )
        fig, ax = plt.subplots()
        im = ax.imshow(np.dstack(self.X), interpolation=interpolation)
        ax.axis("off")
        ax.set_title("Lenia Multi-Channel")

        def __update_multi(i):
            nonlocal im
            self.X = self.filtre.evolve_lenia(self.X)
            im.set_array(np.dstack([self.X[1], self.X[2], self.X[0]]))
            # a choisir entre les deux styles d'affichage:
            # im.set_array(np.dstack(self.X))
            return (im,)

        ani = animation.FuncAnimation(
            fig, __update_multi, frames=num_steps, interval=50, blit=False
        )
        plt.show()

    ### 3 fois la même fonction TODO: à factoriser plus tard ###
    def place_aquarium(self, board: Board, cells: np.ndarray, x: int, y: int):
        h, w = cells.shape[1], cells.shape[2]

        for c in range(cells.shape[0]):
            board.board[x : x + h, y : y + w, c] = cells[c]

    def place_emitter(self, board: Board, cells: np.ndarray, x: int, y: int):
        h, w = cells.shape[1], cells.shape[2]

        for c in range(cells.shape[0]):
            board.board[x : x + h, y : y + w, c] = cells[c]

    def place_pacman(self, board: Board, cells: np.ndarray, x: int, y: int):
        h, w = cells.shape[1], cells.shape[2]

        for c in range(cells.shape[0]):
            board.board[x : x + h, y : y + w, c] = cells[c]
