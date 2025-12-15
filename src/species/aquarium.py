from .abstract_species.A_species import ASpecies
import numpy as np
from typing import Any, Dict, List
from const.constantes import (
    AQUARIUM_T, AQUARIUM_R, AQUARIUM_CELLS, AQUARIUM_KERNEL
)
from scipy.ndimage import shift


class Aquarium(ASpecies):
    def __init__(self) -> None:
        super().__init__()
        self.name = "aquarium"
        self.r: float | None = AQUARIUM_R
        self.t: float | None = AQUARIUM_T
        self.m: float | None = None
        self.s: float | None = None
        self.b: List[float] | None = None
        self.cells: np.ndarray | None = AQUARIUM_CELLS
        self.kernel: Dict[str, Any] | None = AQUARIUM_KERNEL

    def make_patch(
        self,
        rotate: int = 0,
        amplitude: float = 1.0,
        normalize: bool = True,
    ) -> np.ndarray:
        arr = np.asarray(self.cells, dtype=float)  # (h, w, C)

        # rotation SPATIALE uniquement
        if rotate % 4 != 0:
            arr = np.rot90(arr, -(rotate % 4), axes=(0, 1))

        if normalize:
            mx = arr.max()
            if mx > 0:
                arr /= mx

        # barycentre SPATIAL UNIQUEMENT
        yy, xx = np.indices(arr.shape[:2])
        mass = arr.sum()

        if mass > 0:
            cy = (yy[..., None] * arr).sum() / mass
            cx = (xx[..., None] * arr).sum() / mass
        else:
            cy = (arr.shape[0] - 1) / 2
            cx = (arr.shape[1] - 1) / 2

        gy = (arr.shape[0] - 1) / 2
        gx = (arr.shape[1] - 1) / 2

        dy = gy - cy
        dx = gx - cx

        # shift SPATIAL uniquement
        arr = shift(
            arr,
            shift=(dy, dx, 0),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

        return arr * amplitude
