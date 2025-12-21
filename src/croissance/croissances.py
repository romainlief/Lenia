import numpy as np
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

    def choose_growth_function(self, x: np.ndarray) -> np.ndarray:
        """
        If statement to choose the growth function

        Args:
            x (np.ndarray): The input array.

        Raises:
            ValueError: If the type is not in the if statement.

        Returns:
            np.ndarray: The array of values.
        """
        if self.type == Growth_type.GAUSSIENNE:
            if sigma is None or mu is None:
                raise ValueError("Les paramètres sigma et mu ne doivent pas être None.")
            return self.gaussienne(x, sigma=sigma, mu=mu)
        else:
            raise ValueError("Type de fonction de croissance non reconnu.")

    def gaussienne(self, x: np.ndarray, sigma: float, mu: float):
        """
        A Gaussian function but with a return between -1 and 1

        Args:
            x (np.ndarray): input array
            sigma (float): standard deviation
            mu (float): mean

        Returns:
            np.ndarray: output array
        """
        return -1 + 2 * self.gauss(x, mu=mu, sigma=sigma)  # Y de -1 à 1

    def gauss(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """
        A normalized Gaussian function.

        Args:
            x (np.ndarray): Input array.
            mu (float): Mean of the Gaussian.
            sigma (float): Standard deviation of the Gaussian.
        """
        return np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    def target(self, x: np.ndarray, m: float, s: float, A) -> np.ndarray:
        """
        The target function for Wanderer.

        Args:
            x (np.ndarray): Input array.
            m (float): Mean parameter.
            s (float): Standard deviation parameter.
            A (float | None): Additional parameter for the function.
        """
        if A is None:
            return np.exp(-(((x - m) / s) ** 2) / 2)
        else:
            return np.exp(-(((x - m) / s) ** 2) / 2) - A

    def gauss_kernel(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """
        The Gaussian kernel used in Lenia original.

        Args:
            x (np.ndarray): Input array.
            mu (float): Mean of the Gaussian.
            sigma (float): Standard deviation of the Gaussian.
        """
        return np.exp(-(((x - mu) / sigma) ** 2) / 2)

    def bell_growth(self, x: np.ndarray, m: float, s: float, A: float | None = None):
        """
        A bell growth function.

        Args:
            x (np.ndarray): Input array.
            m (float): Mean parameter.
            s (float): Standard deviation parameter.
            A (float | None): Additional parameter for the function. Defaults to None.

        Returns:
            np.ndarray: The bell growth output array.
        """
        return np.exp(-(((x - m) / s) ** 2) / 2) * 2 - 1

    def soft_clip(self, x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """
        A soft clip function.

        Args:
            x (np.ndarray): Input array
            vmin (float): The minimum value
            vmax (float): The maximum value

        Returns:
            np.ndarray: The soft clipped output array
        """
        return 1 / (1 + np.exp(-4 * (x - 0.5)))
