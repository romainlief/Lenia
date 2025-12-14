import numpy as np
from abc import ABC
from typing import Any, Dict, List
from abc import abstractmethod
from scipy.ndimage import shift

class ASpecies(ABC):
    def __init__(self) -> None:
        self.name: str | None
        self.r: float | None
        self.t: float | None
        self.m: float | None
        self.s: float | None
        self.b: List[float] | None
        self.cells: np.ndarray | None
        self.kernel: Dict[str, Any] | None

    def make_patch(
        self,
        rotate: int = 0,
        amplitude: float = 1.0,
        normalize: bool = False,
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

