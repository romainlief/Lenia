import torch
from const.constantes import sigma, mu
from croissance.type_croissance import Growth_type


class Statistical_growth_function:
    """
    Class that store all the statistical function we need to run Lenia.
    """

    def __init__(self, type: Growth_type) -> None:
        """
        Init the Statistical_growth_function class.

        Args:
            type (Type_de_croissance): Type of growth function to use.
        """
        self.type = type

    def choose_growth_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        If statement to choose the growth function

        Args:
            x (torch.Tensor): The input array.

        Raises:
            ValueError: If the type is not in the if statement.

        Returns:
            torch.Tensor: The array of values.
        """
        if self.type == Growth_type.GAUSSIENNE:
            if sigma is None or mu is None:
                raise ValueError("Les paramètres sigma et mu ne doivent pas être None.")
            return self.gaussienne(x, sigma=sigma, mu=mu)
        else:
            raise ValueError("Type de fonction de croissance non reconnu.")

    def gaussienne(self, x: torch.Tensor, sigma: float, mu: float):
        """
        A Gaussian function but with a return between -1 and 1

        Args:
            x (torch.Tensor): input array
            sigma (float): standard deviation
            mu (float): mean

        Returns:
            torch.Tensor: output array
        """
        return -1 + 2 * self.gauss(x, mu=mu, sigma=sigma)  # Y de -1 à 1

    def gauss(self, x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
        """
        A normalized Gaussian function.

        Args:
            x (torch.Tensor): Input array.
            mu (float): Mean of the Gaussian.
            sigma (float): Standard deviation of the Gaussian.
        """
        return torch.exp(-((x - mu) ** 2) / (2 * sigma**2))
    def target(self, x: torch.Tensor, m: float, s: float, A) -> torch.Tensor:
        """
        The target function for Wanderer.

        Args:
            x (torch.Tensor): Input array.
            m (float): Mean parameter.
            s (float): Standard deviation parameter.
            A (float | None): Additional parameter for the function.
        """
        if A is None:
            return torch.exp(-(((x - m) / s) ** 2) / 2)
        else:
            return torch.exp(-(((x - m) / s) ** 2) / 2) - A

    def gauss_kernel(self, x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
        """
        The Gaussian kernel used in Lenia original.

        Args:
            x (torch.Tensor): Input array.
            mu (float): Mean of the Gaussian.
            sigma (float): Standard deviation of the Gaussian.
        """
        return torch.exp(-(((x - mu) / sigma) ** 2) / 2)
    
    def bell_growth(self, x: torch.Tensor, m: float, s: float, A: float | None = None):
        """
        A bell growth function.

        Args:
            x (torch.Tensor): Input array.
            m (float): Mean parameter.
            s (float): Standard deviation parameter.
            A (float | None): Additional parameter for the function. Defaults to None.

        Returns:
            torch.Tensor: The bell growth output array.
        """
        return torch.exp(-(((x - m) / s) ** 2) / 2) * 2 - 1

    def soft_clip(self, x: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
        """
        A soft clip function.

        Args:
            x (torch.Tensor): Input array
            vmin (float): The minimum value
            vmax (float): The maximum value

        Returns:
            torch.Tensor: The soft clipped output array
        """
        return 1 / (1 + torch.exp(-4 * (x - 0.5)))