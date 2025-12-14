from .abstract_species.A_species import ASpecies
import numpy as np
from typing import Any, Dict, List
from const.constantes import (
    FISH_KERNEL, FISH_CELLS, FISH_R, FISH_T
)


class Fish(ASpecies):
    def __init__(self) -> None:
        super().__init__()
        self.name = "fish"
        self.r: float | None = FISH_R
        self.t: float | None = FISH_T
        self.m: float | None = None
        self.s: float | None = None
        self.b: List[float] | None = None
        self.cells: np.ndarray | None = FISH_CELLS
        self.kernel: Dict[str, Any] | None = FISH_KERNEL

    def make_patch(
        self,
        rotate: int = 0,
        amplitude: float = 1.0,
        normalize: bool = True,
    ) -> np.ndarray:
        return super().make_patch(rotate, amplitude, normalize)
