from croissance.croissances import Fonction_de_croissance
from const.constantes import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from croissance.type_croissance import Type_de_croissance


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
        self.world_size = (BOARD_SIZE, BOARD_SIZE)

        if mus is not None and sigmas is not None:
            assert len(mus) == len(sigmas)

        self.R = size // 2
        self.y, self.x = np.ogrid[-self.R : self.R + 1, -self.R : self.R + 1]
        self.distance = np.sqrt(self.x * self.x + self.y * self.y)
        self.r = self.distance / self.R

        self.K = np.zeros_like(self.r)

    def bell(self, x, m, s):
        """Gaussian bell: exp(-((x-m)/s)^2 / 2)"""
        return np.exp(-(((x - m) / s) ** 2) / 2)

    def filtrer(self) -> np.ndarray:
        # kernel local
        K_local = np.zeros_like(self.distance)

        if self.b is not None:
            b_arr = np.asarray(self.b)
            D = self.distance / self.R * len(b_arr)
            ring_indices = np.minimum(D.astype(int), len(b_arr) - 1)
            amplitudes = b_arr[ring_indices]

            K_local = amplitudes * self.bell(D % 1, 0.5, 0.15)
            K_local[D >= len(b_arr)] = 0

        elif self.mus is not None and self.sigmas is not None:
            for mu, sigma in zip(self.mus, self.sigmas):
                K_local += self.fonction_de_croissance.gauss_kernel(self.r, mu, sigma)
            K_local[self.r > 1] = 0
        else:
            raise ValueError("Soit b, soit mus/sigmas doivent être fournis")

        # normalisation locale
        s = K_local.sum()
        if s > 0:
            K_local /= s

        # padding au format de la grille
        H, W = self.world_size
        K = np.zeros((H, W), dtype=float)

        kh, kw = K_local.shape
        cy, cx = kh // 2, kw // 2

        K[:kh, :kw] = K_local
        K = np.roll(K, -cy, axis=0)
        K = np.roll(K, -cx, axis=1)

        return K

    def evolve_lenia(self, X: np.ndarray) -> np.ndarray:
        K = self.filtrer()
        U = np.real(np.fft.ifft2(np.fft.fft2(X) * np.fft.fft2(K)))
        X = X + DT * self.fonction_de_croissance.gaussienne(U, SIGMA, MU)
        return np.clip(X, 0, 1)


# # plot du kernel
# filter = Filtre(
#     fonction_de_croissance=Fonction_de_croissance(type=Type_de_croissance.GAUSSIENNE),
#     size=FILTRE_SIZE,
#     mus=MUS,
#     sigmas=SIGMAS,
# )
# K = filter.filtrer()
# plt.imshow(K, cmap="inferno")
# plt.colorbar()
# plt.title("Kernel Lenia")
# plt.show()
