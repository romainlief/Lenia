import numpy as np
from matplotlib import pyplot as plt
from type_croissance import Type_de_croissance
from constantes import *


class Fonction_de_croissance:
    def __init__(self, type: Type_de_croissance) -> None:
        self.type = type

    def choix_fonction(self, x: np.ndarray) -> np.ndarray | None:
        if self.type == Type_de_croissance.GAUSSIENNE:
            return self.gaussienne(x, sigma=SIGMA, mu=MU)
        else:
            raise ValueError("Type de fonction de croissance non reconnu.")

    def gaussienne(self, x: np.ndarray, sigma: float, mu: float):
        return -1 + 2 * self.gauss(x, mu=mu, sigma=sigma)  # Y de -1 à 1

    def gauss(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return np.exp(-((x - mu) ** 2) / (2 * sigma**2))
