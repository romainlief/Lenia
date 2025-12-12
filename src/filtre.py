from croissance import Fonction_de_croissance
from constantes import *
import numpy as np
import scipy as sp


class Filtre:
    def __init__(
        self,
        fonction_de_croissance: Fonction_de_croissance,
        size: int,
        mus: list,
        sigmas: list,
    ) -> None:
        self.fonction_de_croissance = fonction_de_croissance
        self.size = size
        self.mus = mus 
        self.sigmas = sigmas
        assert len(mus) == len(sigmas)
        self.R = size // 2
        self.y, self.x = np.ogrid[-self.R : self.R + 1, -self.R : self.R + 1]
        self.r = (
            np.sqrt(self.x * self.x + self.y * self.y) / self.R
        ) 

        self.K = np.zeros_like(self.r)

    def filtrer(self) -> np.ndarray:
        for mu, sigma in zip(self.mus, self.sigmas):
            self.K += self.fonction_de_croissance.gauss(self.r, mu, sigma)
        self.K[self.r > 1] = 0  # on met à 0 au delà du rayon
        self.K /= np.sum(self.K)  # normalisation
        return self.K

    def evolve_lenia(self, X: np.ndarray) -> np.ndarray:
        K_lenia = self.filtrer()
        U: np.ndarray = sp.signal.convolve2d(X, K_lenia, mode="same", boundary="wrap")
        X = X + DT * self.fonction_de_croissance.gaussienne(U, SIGMA, MU)
        X = np.clip(X, 0, 1)
        return X
