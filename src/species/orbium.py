import numpy as np
from typing import Any, Dict, List
from const.constantes import (
    ORBIUM_M,
    ORBIUM_S,
    ORBIUM_T,
    ORBIUM_R,
    ORBIUM_B,
    ORBIUM_CELLS,
)
from scipy.ndimage import shift
from .abstract_species.A_species import ASpecies


class Orbium(ASpecies):
    def __init__(self) -> None:
        super().__init__()
        self.name = "orbium"
        self.r: float | None = ORBIUM_R
        self.t: float | None = ORBIUM_T
        self.m: float | None = ORBIUM_M
        self.s: float | None = ORBIUM_S
        self.b: List[float] | None = ORBIUM_B
        self.cells: np.ndarray | None = ORBIUM_CELLS
        self.kernel: Dict[str, Any] | None = None

    def make_patch(
        self,
        rotate: int = 0,
        amplitude: float = 1.0,
        normalize: bool = True,
    ) -> np.ndarray:
        arr = np.asarray(self.cells, dtype=float)

        if rotate % 4 != 0:
            arr = np.rot90(arr, -(rotate % 4))

        if normalize:
            mx = arr.max()
            if mx > 0:
                arr = arr / mx
        # centre de masse CONTINU
        yy, xx = np.indices(arr.shape)
        mass = arr.sum()

        if mass > 0:
            cy = (yy * arr).sum() / mass
            cx = (xx * arr).sum() / mass
        else:
            cy = (arr.shape[0] - 1) / 2
            cx = (arr.shape[1] - 1) / 2

        # centre géométrique continu
        gy = (arr.shape[0] - 1) / 2
        gx = (arr.shape[1] - 1) / 2
        # décalage CONTINU (clé Lenia)
        dy = gy - cy
        dx = gx - cx

        # interpolation bilinéaire (order=1)
        arr = shift(
            arr,
            shift=(dy, dx),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

        return arr * amplitude
