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
        return super().make_patch(rotate, amplitude, normalize)
