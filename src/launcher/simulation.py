from space.box_space import BoxSpace
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
    kernel_type,
    USE_HYDROGEMINIUM_PARAMS,
    USE_ORBIUM_PARAMS,
    USE_FISH_PARAMS,
    USE_WANDERER_PARAMS,
    channel_count,
    VOID_BOARD,
)
from species.orbium import Orbium
from species.hydrogeminium import Hydrogeminium
from species.fish import Fish
from species.wanderer import Wanderer
from croissance.type_croissance import Growth_type

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.image import AxesImage
from matplotlib.figure import Figure
from scipy.ndimage import shift
from species.species_types import Species_types
import torch


class Simulation:
    def __init__(
        self,
        size: int = BOARD_SIZE,
        kernel_type: Species_types = kernel_type,
        channel_count: int = channel_count,
    ) -> None:
        """
        Initialize the simulation.
        size        : Size of the board (size x size)
        kernel_type : Type of species kernel to use
        """
        self.size = size
        self.channel_count = channel_count

        self.space = BoxSpace(
            low=0.0,
            high=1.0,
            shape=(size, size, channel_count),
            dtype=torch.float32,
        )
        if VOID_BOARD:
            self.X_raw = torch.zeros(self.space.shape, dtype=torch.float32)
        else:
            self.X_raw = torch.rand(self.space.shape, dtype=torch.float32)

        self.multi_channel: bool = self.channel_count > 1

        if USE_AQUARIUM_PARAMS:
            self.place_multi_chan_species(
                self.X_raw,
                AQUARIUM_CELLS,
                self.X_raw.shape[0] // 2,
                self.X_raw.shape[1] // 2,
            )

        if USE_EMITTER_PARAMS:
            self.place_multi_chan_species(
                self.X_raw,
                EMITTER_CELLS,
                self.X_raw.shape[0] // 2,
                self.X_raw.shape[1] // 2,
            )

        if USE_PACMAN_PARAMS:
            self.place_multi_chan_species(
                self.X_raw,
                PACMAN_CELLS,
                self.X_raw.shape[0] // 2,
                self.X_raw.shape[1] // 2,
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

        # Représentation utilisée pour l'évolution : soit 2D, soit liste de plans (multi-canaux)
        self.x: torch.Tensor | list[torch.Tensor]
        if self.X_raw.ndim == 3 and self.multi_channel:
            self.x = [self.X_raw[:, :, c].clone() for c in range(self.X_raw.shape[2])]
        elif self.X_raw.ndim == 3:
            self.x = torch.mean(self.X_raw, axis=2)
        else:
            self.x = self.X_raw.clone()
        patch = None
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
        if (
            not USE_AQUARIUM_PARAMS
            and not USE_EMITTER_PARAMS
            and not USE_PACMAN_PARAMS
            and patch is not None
        ):
            self.apply_patch(patch, center=(BOARD_SIZE // 2, BOARD_SIZE // 2))

    def apply_patch(
        self,
        patch: torch.Tensor | None = None,
        center: tuple[int, int] | None = None,
        rotate: int = 0,
        amplitude: float = 1.0,
        normalize: bool = True,
    ) -> None:
        """
        Apply a 2D patch to the grid self.x (Torch tensors, multi-channel supported).

        patch     : The 2D torch.Tensor to apply as a patch
        center    : (y,x) coordinates of the center where to place the patch
        rotate    : number of 90° rotations (0..3)
        amplitude : multiplicative factor applied to the patch
        normalize : if True, normalize the patch to 1 before applying amplitude
        """
        if patch is None:
            return

        # Convert patch to torch tensor
        arr = torch.tensor(patch, dtype=self.X_raw.dtype, device=self.X_raw.device)
        if rotate % 4 != 0:
            arr = torch.rot90(arr, k=-(rotate % 4))

        if normalize:
            mx = arr.max()
            if mx > 0:
                arr /= mx

        # Recenter patch
        yy, xx = torch.meshgrid(
            torch.arange(arr.shape[0], device=arr.device),
            torch.arange(arr.shape[1], device=arr.device),
            indexing="ij",
        )
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
            arr.cpu().numpy(),
            shift=(dy.item(), dx.item()),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        arr = torch.tensor(arr, dtype=self.X_raw.dtype, device=self.X_raw.device)

        arr *= amplitude

        h, w = arr.shape
        if center is None:
            cy, cx = self.size // 2, self.size // 2
        else:
            cy, cx = center

        top = int(round(cy - h / 2))
        left = int(round(cx - w / 2))

        # Apply patch with wrapping
        for dy in range(h):
            for dx in range(w):
                y = (top + dy) % self.size
                x = (left + dx) % self.size
                if isinstance(self.x, list):
                    for c in range(len(self.x)):
                        self.x[c][y, x] = torch.clamp(
                            self.x[c][y, x] + arr[dy, dx], 0.0, 1.0
                        )
                else:
                    self.x[y, x] = torch.clamp(self.x[y, x] + arr[dy, dx], 0.0, 1.0)

        # synchronize the raw multi-channel grid
        if hasattr(self, "X_raw") and self.X_raw.ndim == 3:
            if isinstance(self.x, list):
                self.X_raw = torch.clamp(torch.stack(self.x, dim=2), 0.0, 1.0)
            else:
                self.X_raw = torch.clamp(
                    torch.stack([self.x] * self.X_raw.shape[2], dim=2), 0.0, 1.0
                )

    def __update(self, frame: int) -> list[AxesImage]:
        # Convert multi-channel list to single-channel tensor
        if isinstance(self.x, list):
            x_input = torch.mean(torch.stack(self.x, dim=0), dim=0)
        else:
            x_input = self.x

        # s'assurer que x_input est torch.Tensor
        if not isinstance(x_input, torch.Tensor):
            x_input = torch.tensor(x_input, dtype=torch.float32)

        # Appeler evolve_lenia en convertissant les sorties en torch.Tensor
        result = self.filtre.evolve_lenia(x_input)
        print(type(result))
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result, dtype=torch.float32)
        elif isinstance(result, list):
            result = [
                (
                    torch.tensor(r, dtype=torch.float32)
                    if not isinstance(r, torch.Tensor)
                    else r
                )
                for r in result
            ]

        # Mettre à jour self.x
        if isinstance(result, list):
            self.x = [r.clone() for r in result]
            display = torch.mean(torch.stack(self.x, dim=0), dim=0).cpu().numpy()
        else:
            self.x = result
            display = self.x.cpu().numpy()

        self.img.set_data(display)
        return [self.img]

    def _enable_resize(self, fig: Figure) -> None:
        """
        Attach a resize handler so the figure redraws and layout updates when
        the window is resized (useful for GUI backends).

        Args:
            fig (Figure): The matplotlib figure to attach the resize handler to.
        """

        def _on_resize(event):
            """
            Handle the resize event for the figure.
            """
            try:
                fig.tight_layout()
            except Exception:
                pass  # tight_layout can fail in some edge cases; ignore silently
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("resize_event", _on_resize)

    def zoom_in(self, fig: Figure, factor: float = 1.1) -> None:
        """
        Zoom in the current view of the figure.

        Args:
            factor (float, optional): The zoom factor. Defaults to 1.2.
            fig (plt.Figure): The matplotlib figure to zoom in.
        """

        def _on_zoom(event):
            """
            Handle the zoom event for the figure.
            """
            ax = fig.axes[0]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            x_range = (xlim[1] - xlim[0]) / factor
            y_range = (ylim[1] - ylim[0]) / factor
            ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
            ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
            fig.canvas.draw_idle()

        def _zoom_out(event):
            """
            Handle the zoom out event for the figure.
            """
            ax = fig.axes[0]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            x_range = (xlim[1] - xlim[0]) * factor
            y_range = (ylim[1] - ylim[0]) * factor
            ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
            ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
            fig.canvas.draw_idle()

        def _on_scroll(event):
            """
            Handle the scroll event for zooming in and out.
            """
            if event.button == "up":
                _on_zoom(event)
            elif event.button == "down":
                _zoom_out(event)

        fig.canvas.mpl_connect("scroll_event", _on_scroll)

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
        self.img = ax.imshow(self.x, cmap="inferno", interpolation="none")
        ax.set_title("Lenia")
        ax.set_xticks([])
        ax.set_yticks([])
        anim = animation.FuncAnimation(
            fig, self.__update, frames=200, interval=20, blit=True
        )
        self._enable_resize(fig)
        self.zoom_in(fig=fig)
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
        if not isinstance(self.x, list):
            raise RuntimeError(
                "run_multi requires multi-channel board (self.X as list)"
            )
        fig, ax = plt.subplots()
        self.x = [
            torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x for x in self.x
        ]
        im = ax.imshow(
            torch.stack(self.x, dim=2).cpu().numpy(), interpolation=interpolation
        )
        ax.axis("off")
        ax.set_title("Lenia Multi-Channel")

        def __update_multi(i):
            """
            The update function for multi channel species
            """
            nonlocal im
            result = self.filtre.evolve_lenia(self.x)  # type: ignore
            # S'assurer que result est une liste de torch.Tensor
            if not isinstance(result, list):
                if isinstance(result, torch.Tensor):
                    self.x = [result] if result.ndim == 2 else list(result)
                else:
                    self.x = [torch.as_tensor(result)]
            else:
                self.x = [
                    torch.as_tensor(r) if not isinstance(r, torch.Tensor) else r
                    for r in result
                ]
            # Affichage RGB : conversion en numpy pour matplotlib

            rgb = torch.stack([self.x[1], self.x[2], self.x[0]], dim=2).cpu().numpy()
            #  rgb =  torch.dstack(self.x)
            im.set_array(rgb)
            return (im,)

        ani = animation.FuncAnimation(
            fig, __update_multi, frames=num_steps, interval=50, blit=False
        )
        self._enable_resize(fig)
        self.zoom_in(fig=fig)
        plt.show()

    def place_multi_chan_species(
        self, space: torch.Tensor, cells: torch.Tensor, x: int, y: int
    ):
        """
        Place multi-channel species on the board.

        Args:
            space (torch.Tensor): The board space where to place the species.
            cells (torch.Tensor): The multi-channel species cells to place.
            x (int): The x-coordinate on the board.
            y (int): The y-coordinate on the board.
        """
        h, w = cells.shape[1], cells.shape[2]
        H, W = space.shape[0], space.shape[1]

        for c in range(cells.shape[0]):
            for dy in range(h):
                for dx in range(w):
                    yy = (x + dy) % H
                    xx = (y + dx) % W
                    space[yy, xx, c] = torch.clamp(
                        space[yy, xx, c] + cells[c, dy, dx], 0.0, 1.0
                    )
