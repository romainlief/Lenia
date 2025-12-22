import torch
from abc import ABC
from typing import Any, Dict, List
from scipy.ndimage import shift

class ASpecies(ABC):
    """
    Abstract Species class.
    """
    def __init__(self) -> None:
        """
        Initialise the abstract species with default attributes.
        """
        self.name: str | None
        self.r: float | None
        self.t: float | None
        self.m: float | None
        self.s: float | None
        self.b: List[float] | None
        self.cells: torch.Tensor | None
        self.kernel: Dict[str, Any] | None

    def make_patch(
        self,
        rotate: int = 0,
        amplitude: float = 1.0,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Create a patch of the species with specified rotation and amplitude.
        
        Args:
            rotate (int, optional): Rotation angle in multiples of 90 degrees. Defaults to 0.
            amplitude (float, optional): Amplitude scaling factor. Defaults to 1.0.
            normalize (bool, optional): Whether to normalize the patch. Defaults to False.

        Returns:
            torch.Tensor: The generated patch array.
        """
        arr = torch.tensor(self.cells, dtype=torch.float)
        

        if rotate % 4 != 0:
            arr = torch.rot90(arr, -(rotate % 4))
        if normalize:
            mx = arr.max()
            if mx > 0:
                arr = arr / mx
        # Continuous center of mass
        yy, xx = torch.meshgrid(torch.arange(arr.shape[0]), torch.arange(arr.shape[1]), indexing="ij")
        mass = arr.sum()

        if mass > 0:
            cy = (yy * arr).sum() / mass
            cx = (xx * arr).sum() / mass
        else:
            cy = (arr.shape[0] - 1) / 2
            cx = (arr.shape[1] - 1) / 2

        # Continuous geometric center
        gy = (arr.shape[0] - 1) / 2
        gx = (arr.shape[1] - 1) / 2
        # Continuous offset (Lenia key feature)
        dy = gy - cy
        dx = gx - cx

        # Bilinear interpolation (order=1)
        arr = shift(
            arr,
            shift=(dy, dx),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

        return arr * amplitude

