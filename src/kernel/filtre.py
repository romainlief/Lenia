import numpy as np
from const.constantes import *
from croissance.croissances import Fonction_de_croissance
from croissance.type_croissance import Type_de_croissance
from species.species_types import Species_types
from scipy.ndimage import zoom
class Filtre:
    def __init__(
        self,
        fonction_de_croissance: Fonction_de_croissance,
        size: int,
        mus: list | None = None,
        sigmas: list | None = None,
        b: list | None = None,
        kernels: list[dict] | None = None,
        multi_channel: bool = False,
        species_type: Species_types | None = None,
    ) -> None:
        self.fonction_de_croissance = fonction_de_croissance
        self.size = size
        self.mus = mus
        self.sigmas = sigmas
        self.b = b
        self.kernels = kernels
        self.world_size = (BOARD_SIZE, BOARD_SIZE)
        self.R = size // 2
        self.y, self.x = np.ogrid[-self.R : self.R + 1, -self.R : self.R + 1]
        self.distance = np.sqrt(self.x**2 + self.y**2)
        self.r = self.distance / self.R
        self.multi_channel = multi_channel
        self.species_type = species_type

        if mus is not None and sigmas is not None:
            assert len(mus) == len(sigmas)

        # Préparation des kernels FFT dès l'init pour Fish
        self.prepared_kernels_fft = None
        if self.kernels is not None:
            self.prepared_kernels_fft = self.prepare_fish_kernels_fft()

    def bell(self, x, m, s):
        """Gaussian bell: exp(-((x-m)/s)^2 / 2)"""
        return np.exp(-(((x - m) / s) ** 2) / 2)

    def prepare_fish_kernels_fft(self):
        """Prépare les kernels FFT pour Fish (multi-kernels)"""
        H, W = self.world_size
        mid_h, mid_w = H // 2, W // 2
        kernels_fft = []
        for k in self.kernels:
            y, x = np.ogrid[-mid_h:mid_h, -mid_w:mid_w]
            # radius for this kernel (relative factor in pattern)
            r_k = max(1e-6, self.R * k.get("r", 1.0))
            # distance scaled by kernel radius and number of rings
            D = np.sqrt(x**2 + y**2) / r_k * len(k["b"])
            amplitudes = np.asarray(k["b"])[np.minimum(D.astype(int), len(k["b"]) - 1)]
            K_local = amplitudes * self.bell(D % 1, 0.5, 0.15)
            K_local[D >= len(k["b"])] = 0
            K_local /= K_local.sum() if K_local.sum() > 0 else 1
            kernels_fft.append(np.fft.fft2(np.fft.fftshift(K_local)))

        return kernels_fft

    def filtrer(self):
        """Retourne le kernel FFT pour Lenia classique ou liste de FFT pour Fish"""
        # Mode Fish
        if self.kernels is not None:
            return self.prepared_kernels_fft

        # Mode Lenia classique
        H, W = self.world_size
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

        # Normalisation
        K_local /= K_local.sum() if K_local.sum() > 0 else 1

        # Padding au format de la grille
        K = np.zeros((H, W), dtype=float)
        kh, kw = K_local.shape
        cy, cx = kh // 2, kw // 2
        K[:kh, :kw] = K_local
        K = np.roll(K, -cy, axis=0)
        K = np.roll(K, -cx, axis=1)

        return np.fft.fft2(K)

    def bell_growth(self, U, m, s):
        """Croissance pour Fish"""
        return self.bell(U, m, s) * 2 - 1

    def evolve_lenia(self, X: np.ndarray):
        # Mode Fish
        if self.kernels is not None and not self.multi_channel and self.species_type == Species_types.FISH:
            Ks = self.prepared_kernels_fft
            Us = [np.real(np.fft.ifft2(fK * np.fft.fft2(X))) for fK in Ks]
            Gs = [self.bell_growth(U, k["m"], k["s"]) for U, k in zip(Us, self.kernels)]
            X = np.clip(X + DT * np.mean(np.asarray(Gs), axis=0), 0, 1)

        elif (  # Mode Aquarium
            self.kernels is not None and self.multi_channel and self.species_type == Species_types.AQUARIUM
        ):  # Multi-canaux pour Aquarium
            fXs = [np.fft.fft2(Xi) for Xi in X]
            Gs = [np.zeros_like(Xi) for Xi in X]

            for i, (fK, src, dst, h) in enumerate(
                zip(
                    self.prepared_kernels_fft,
                    SOURCE_AQUARIUM,
                    DESTINATION_AQUARIUM,
                    AQUARIUM_H,
                )
            ):
                U = np.real(np.fft.ifft2(fK * fXs[src]))
                A = self.bell_growth(U, AQUARIUM_ms[i], AQUARIUM_ss[i])
                Gs[dst] += h * A

            return [np.clip(Xi + DT * Gi, 0, 1) for Xi, Gi in zip(X, Gs)]
        else:  # Lenia classique
            K = self.filtrer()
            U = np.real(np.fft.ifft2(np.fft.fft2(X) * K))
            if self.species_type == Species_types.WANDERER:
                target = np.exp(-(((U - WANDERER_M) / WANDERER_S) ** 2) / 2)
                X = np.clip(X + DT * (target - X), 0, 1)
            else:
                X = X + DT * self.fonction_de_croissance.gaussienne(U, SIGMA, MU)
                X = np.clip(X, 0, 1)
        return X
