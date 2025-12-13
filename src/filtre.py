from croissance import Fonction_de_croissance
from constantes import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from type_croissance import Type_de_croissance


class Filtre:
    def __init__(
        self,
        fonction_de_croissance: Fonction_de_croissance,
        size: int,
        mus: list | None = None,
        sigmas: list | None = None,
        b: list | None = None,
    ) -> None:
        self.fonction_de_croissance = fonction_de_croissance
        self.size = size
        self.mus = mus
        self.sigmas = sigmas
        self.b = b  # pour kernel multi-anneau
        
        if mus is not None and sigmas is not None:
            assert len(mus) == len(sigmas)
        
        self.R = size // 2
        self.y, self.x = np.ogrid[-self.R : self.R + 1, -self.R : self.R + 1]
        self.distance = np.sqrt(self.x * self.x + self.y * self.y)
        self.r = self.distance / self.R

        self.K = np.zeros_like(self.r)

    def bell(self, x, m, s):
        """Gaussian bell: exp(-((x-m)/s)^2 / 2)"""
        return np.exp(-((x - m) / s) ** 2 / 2)

    def filtrer(self) -> np.ndarray:
        """Construit le kernel. Utilise soit b (multi-anneau) soit MUS/SIGMAS."""
        if self.b is not None:
            # Approche multi-anneau (comme le code référence)
            b_arr = np.asarray(self.b)
            D = self.distance / self.R * len(b_arr)  # Normaliser en "unités d'anneau"
            
            # Sélectionner l'amplitude pour chaque anneau
            ring_indices = np.minimum(D.astype(int), len(b_arr) - 1)
            amplitudes = b_arr[ring_indices]
            
            # Appliquer le gaussian lisse à l'intérieur de chaque anneau
            # D % 1 = partie fractionnelle (0-1 dans chaque anneau)
            self.K = amplitudes * self.bell(D % 1, 0.5, 0.15)
            
            # Masquer au-delà du rayon
            self.K[D >= len(b_arr)] = 0
        elif self.mus is not None and self.sigmas is not None:
            for mu, sigma in zip(self.mus, self.sigmas):
                self.K += self.fonction_de_croissance.gauss_kernel(self.r, mu, sigma)
            self.K[self.r > 1] = 0  # on met à 0 au delà du rayon
        else:
            raise ValueError("Soit b, soit mus/sigmas doivent être fournis")
        
        K_sum = np.sum(self.K)
        if K_sum > 0:
            self.K /= K_sum  # normalisation
        return self.K

    def evolve_lenia(self, X: np.ndarray) -> np.ndarray:
        K_lenia = self.filtrer()
        U: np.ndarray = sp.signal.convolve2d(X, K_lenia, mode="same", boundary="wrap")
        X = X + DT * self.fonction_de_croissance.gaussienne(U, SIGMA, MU)
        X = np.clip(X, 0, 1)
        return X


# plot du kernel
filter = Filtre(
    fonction_de_croissance=Fonction_de_croissance(type=Type_de_croissance.GAUSSIENNE),
    size=FILTRE_SIZE,
    mus=MUS,
    sigmas=SIGMAS,
)
K = filter.filtrer()
plt.imshow(K, cmap="inferno")
plt.colorbar()
plt.title("Kernel Lenia")
plt.show()
