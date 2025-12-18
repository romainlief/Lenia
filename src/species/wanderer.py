import numpy as np
from typing import Any, Dict, List
from const.constantes import (
    WANDERER_M,
    WANDERER_S,
    WANDERER_T,
    WANDERER_R,
    WANDERER_B,
    WANDERER_CELLS,
)
from .abstract_species.A_species import ASpecies


class Wanderer(ASpecies):
    def __init__(self) -> None:
        super().__init__()
        self.name = "wanderer"
        self.r: float | None = WANDERER_R
        self.t: float | None = WANDERER_T
        self.m: float | None = WANDERER_M
        self.s: float | None = WANDERER_S
        self.b: List[float] | None = WANDERER_B
        self.cells: np.ndarray | None = WANDERER_CELLS
        self.kernel: Dict[str, Any] | None = None

    def make_patch(
        self,
        rotate: int = 0,
        amplitude: float = 1.0,
        normalize: bool = True,
    ) -> np.ndarray:
        return super().make_patch(rotate, amplitude, normalize)
