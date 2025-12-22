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
from species.species_types import Species_types
import torch
from scipy.ndimage import shift


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
            self.x = torch.mean(self.X_raw, axis=2) # type: ignore
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
            shift=(dy, dx),
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

    def reset(self) -> None:
        """Reset the simulation to the initial state."""
        if self.X_raw is not None:
            if self.X_raw.ndim == 3 and self.multi_channel:
                self.x = [self.X_raw[:, :, c].clone() for c in range(self.X_raw.shape[2])]
            elif self.X_raw.ndim == 3:
                self.x = torch.mean(self.X_raw, axis=2)  # type: ignore
            else:
                self.x = self.X_raw.clone()