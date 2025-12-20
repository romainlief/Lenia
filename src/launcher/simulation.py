from board.board import Board
from croissance.croissances import Statistical_growth_function
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
from croissance.type_croissance import Growth_type

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
        Initialize the simulation.
        size        : Size of the board (size x size)
        kernel_type : Type of species kernel to use
        """
        self.size = size
        self.grid = Board()
        self.multi_channel: bool = self.grid.channels > 1

        if USE_AQUARIUM_PARAMS:
            self.place_multi_chan_species(
                self.grid, AQUARIUM_CELLS, self.grid.size // 2, self.grid.size // 2
            )

        if USE_EMITTER_PARAMS:
            self.place_multi_chan_species(
                self.grid, EMITTER_CELLS, self.grid.size // 2, self.grid.size // 2
            )

        if USE_PACMAN_PARAMS:
            self.place_multi_chan_species(
                self.grid, PACMAN_CELLS, self.grid.size // 4, self.grid.size // 4
            )

        if kernel_type == Species_types.HYDROGEMINIUM:
            b = HYDROGEMINIUM_B
            self.filtre = Filtre(
                growth_function=Statistical_growth_function(
                    type=Growth_type.GAUSSIENNE,
                ),
                size=FILTRE_SIZE,
                b=b,
                species_type=Species_types.HYDROGEMINIUM,
            )
        elif kernel_type == Species_types.ORBIUM:
            b = ORBIUM_B
            self.filtre = Filtre(
                growth_function=Statistical_growth_function(
                    type=Growth_type.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=b,
                species_type=Species_types.ORBIUM,
            )
        elif kernel_type == Species_types.FISH:
            self.filtre = Filtre(
                growth_function=Statistical_growth_function(
                    type=Growth_type.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=None,
                kernels=FISH_KERNEL,
                species_type=Species_types.FISH,
            )
            self.filtre.kernels = FISH_KERNEL
        elif kernel_type == Species_types.AQUARIUM:
            self.filtre = Filtre(
                growth_function=Statistical_growth_function(
                    type=Growth_type.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=None,
                kernels=AQUARIUM_KERNEL,
                multi_channel=self.multi_channel,
                species_type=Species_types.AQUARIUM,
            )
            self.filtre.kernels = AQUARIUM_KERNEL
        elif kernel_type == Species_types.WANDERER:
            b = WANDERER_B
            self.filtre = Filtre(
                growth_function=Statistical_growth_function(
                    type=Growth_type.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=b,
                species_type=Species_types.WANDERER,
            )
        elif kernel_type == Species_types.EMITTER:
            self.filtre = Filtre(
                growth_function=Statistical_growth_function(
                    type=Growth_type.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                b=None,
                kernels=EMITTER_KERNEL,
                multi_channel=self.multi_channel,
                species_type=Species_types.EMITTER,
            )
            self.filtre.kernels = EMITTER_KERNEL 
        elif kernel_type == Species_types.PACMAN:
            self.filtre = Filtre(
                growth_function=Statistical_growth_function(
                    type=Growth_type.GAUSSIENNE
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
                growth_function=Statistical_growth_function(
                    type=Growth_type.GAUSSIENNE
                ),
                size=FILTRE_SIZE,
                mus=MUS,
                sigmas=SIGMAS,
                species_type=Species_types.GENERIC,
            )

        self.X_raw = self.grid.board.copy()
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
        Apply a 2D patch to the grid self.X.

        patch     : The 2D numpy array to apply as a patch
        center    : (y,x) coordinates of the center where to place the patch
        rotate    : number of 90° rotations (0..3)
        amplitude : multiplicative factor applied to the patch
        normalize : if True, normalize the patch to 1 before applying amplitude
        
        Returns:    None
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

        #  recenter
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

        arr *= amplitude

        h, w = arr.shape  # Height and width of the patch
        if center is None:
            cy, cx = self.size // 2, self.size // 2
        else:
            cy, cx = center

        top = int(round(cy - h / 2))
        left = int(round(cx - w / 2))

        # Apply the patch to self.X with wrapping
        for dy in range(h):
            for dx in range(w):
                y = (top + dy) % self.size
                x = (left + dx) % self.size
                if isinstance(self.X, list):
                    for c in range(len(self.X)):
                        self.X[c][y, x] = np.clip(self.X[c][y, x] + arr[dy, dx], 0, 1)
                else:
                    self.X[y, x] = np.clip(self.X[y, x] + arr[dy, dx], 0, 1)

        # synchronize the raw multi-channel grid
        if hasattr(self, "X_raw") and self.X_raw.ndim == 3:
            if isinstance(self.X, list):
                self.X_raw = np.clip(np.stack(self.X, axis=2), 0, 1)
            else:
                self.X_raw = np.clip(
                    np.stack([self.X] * self.X_raw.shape[2], axis=2), 0, 1
                )

    def __update(self, frame: int) -> list:
        """
        Update the simulation for 1 channel board

        Args:
            frame (int): The current frame number.

        Returns:
            list: A list containing the updated image.
        """
        self.X = self.filtre.evolve_lenia(self.X)
        display = self.X
        self.img.set_data(display)
        return [self.img]

    def _enable_resize(self, fig: plt.Figure) -> None:
        """
        Attach a resize handler so the figure redraws and layout updates when
        the window is resized (useful for GUI backends).
        
        Args:
            fig (plt.Figure): The matplotlib figure to attach the resize handler to.
        """
        def _on_resize(event):
            try:
                fig.tight_layout()
            except Exception:
                # tight_layout can fail in some edge cases; ignore silently
                pass
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("resize_event", _on_resize)

    def run(self, num_steps=100, interpolation="bicubic"):
        """
        The method that choose the run the simulation in function of if its multi_channel or not.

        Args:
            num_steps (int, optional): The number of steps to run the simulation. Defaults to 100.
            interpolation (str, optional): The interpolation method for displaying the image. Defaults to "bicubic".
        """
        if self.multi_channel:
            self.__run_multi(num_steps=num_steps, interpolation=interpolation)
        else:
            self.__run()

    def __run(self):
        """
        Run the simulation for single-channel boards.
        """
        fig, ax = plt.subplots()
        self.img = ax.imshow(self.X, cmap="inferno", interpolation="none")
        ax.set_title("Lenia")
        ax.set_xticks([])
        ax.set_yticks([])
        anim = animation.FuncAnimation(
            fig, self.__update, frames=200, interval=20, blit=True
        )
        self._enable_resize(fig)
        plt.show()

    def __run_multi(self, num_steps=100, interpolation="bicubic"):
        """
        Run the simulation for multi-channel boards.
        
        Args:
            num_steps (int, optional): The number of steps to run the simulation. Defaults to 100.
            interpolation (str, optional): The interpolation method for displaying the image. Defaults to "bicubic".

        Raises:
            RuntimeError: If self.X is not a list (i.e., not multi-channel).

        Returns:
            None
        """
        if not isinstance(self.X, list):
            raise RuntimeError(
                "run_multi requires multi-channel board (self.X as list)"
            )
        fig, ax = plt.subplots()
        im = ax.imshow(np.dstack(self.X), interpolation=interpolation)
        ax.axis("off")
        ax.set_title("Lenia Multi-Channel")

        def __update_multi(i):
            """
            The update function for multi channel species

            Returns:
                _type_: _tuple of im
            """
            nonlocal im
            self.X = self.filtre.evolve_lenia(self.X)
            im.set_array(np.dstack([self.X[1], self.X[2], self.X[0]]))
            # a choisir entre les deux styles d'affichage:
            # im.set_array(np.dstack(self.X))
            return (im,)

        ani = animation.FuncAnimation(
            fig, __update_multi, frames=num_steps, interval=50, blit=False
        )
        self._enable_resize(fig)
        plt.show()

    def place_multi_chan_species(self, board: Board, cells: np.ndarray, x: int, y: int):
        """
        Place multi-channel species on the board.
        
        Args:
            board (Board): The board on which to place the species.
            cells (np.ndarray): The multi-channel species cells to place.
            x (int): The x-coordinate on the board.
            y (int): The y-coordinate on the board.
        """
        h, w = cells.shape[1], cells.shape[2]
        for c in range(cells.shape[0]):
            board.board[x : x + h, y : y + w, c] = cells[c]
