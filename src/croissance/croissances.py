import numpy as np
from const.constantes import SIGMA, MU
from croissance.type_croissance import Type_de_croissance


class Fonction_de_croissance:
    def __init__(self, type: Type_de_croissance) -> None:
        self.type = type

    def choix_fonction(self, x: np.ndarray) -> np.ndarray:
        if self.type == Type_de_croissance.GAUSSIENNE:
            return self.gaussienne(x, sigma=SIGMA, mu=MU)
        else:
            raise ValueError("Type de fonction de croissance non reconnu.")

    def gaussienne(self, x: np.ndarray, sigma: float, mu: float):
        return -1 + 2 * self.gauss(x, mu=mu, sigma=sigma)  # Y de -1 à 1

    def gauss(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """Gaussienne normalisée pour la croissance et le kernel."""
        return np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    
    def target(self, x: np.ndarray, m: float, s: float, A=None) -> np.ndarray:
        """Target function for Wanderer."""
        if A is None:
            return np.exp(-(((x - m) / s) ** 2) / 2)
        else:
            print("ici")
            return np.exp(-(((x - m) / s) ** 2) / 2) - A

    def gauss_kernel(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """Gaussienne du kernel (version 'bell' — sans normalisation par écart-type²)."""
        # Dans Lenia original, le kernel est : exp(-((x-mu)/sigma)^2 / 2)
        # Cela donne une cloche plus "serrée" que gauss pour même sigma
        return np.exp(-(((x - mu) / sigma) ** 2) / 2)
    
    def bell_growth(self, U, m, s, A=None):
        return np.exp(-(((U - m) / s) ** 2) / 2) * 2 - 1
    
    def soft_clip(self, x, vmin, vmax):
        return 1 / (1 + np.exp(-4 * (x - 0.5)))
