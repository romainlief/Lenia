import numpy as np
from matplotlib import pyplot as plt
from type_croissance import Type_de_croissance
from constantes import *


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

    def gauss_kernel(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """Gaussienne du kernel (version 'bell' — sans normalisation par écart-type²)."""
        # Dans Lenia original, le kernel est : exp(-((x-mu)/sigma)^2 / 2)
        # Cela donne une cloche plus "serrée" que gauss pour même sigma
        return np.exp(-(((x - mu) / sigma) ** 2) / 2)

    def other_function(self, x: np.ndarray) -> np.ndarray:
        return 0 + ((x >= 0.12) & (x <= 0.15)) - ((x < 0.12) | (x > 0.15))


# plot de la fonction de croissance
fct = Fonction_de_croissance(type=Type_de_croissance.GAUSSIENNE)
x = np.linspace(0, 1, 500)
y = fct.choix_fonction(x)
plt.plot(x, y)
plt.title("Fonction de croissance gaussienne")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()
