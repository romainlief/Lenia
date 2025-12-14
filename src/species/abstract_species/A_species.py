import numpy as np
from abc import ABC
from typing import Any, Dict, List
from abc import abstractmethod


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

    @abstractmethod
    def make_patch(
        self,
        rotate: int = 0,
        amplitude: float = 1.0,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Retourne un patch 2D prêt à être appliqué sur une grille.
        """
        pass
