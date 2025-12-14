from .abstract_species.A_species import ASpecies
import numpy as np
from typing import Any, Dict, List
from const.constantes import (
    HYDROGEMINIUM_M,
    HYDROGEMINIUM_S,
    HYDROGEMINIUM_T,
    HYDROGEMINIUM_R,
    HYDROGEMINIUM_B,
    HYDROGEMINIUM_CELLS,
)
from scipy.ndimage import shift


class Hydrogeminium(ASpecies):
    def __init__(self) -> None:
        super().__init__()
        self.name = "geminium"
        self.r: float | None = HYDROGEMINIUM_R
        self.t: float | None = HYDROGEMINIUM_T
        self.m: float | None = HYDROGEMINIUM_M
        self.s: float | None = HYDROGEMINIUM_S
        self.b: List[float] | None = HYDROGEMINIUM_B
        self.cells: np.ndarray | None = HYDROGEMINIUM_CELLS
        self.kernel: Dict[str, Any] | None = None

    def make_patch(
        self,
        rotate: int = 0,
        amplitude: float = 1.0,
        normalize: bool = True,
    ) -> np.ndarray:
        return super().make_patch(rotate, amplitude, normalize)
