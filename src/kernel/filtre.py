from const import constantes as CONST
from croissance.croissances import Statistical_growth_function
from species.species_types import Species_types
import torch


class Filtre:
    """
    Filtre class to handle convolution and evolution of the world state
    using specified growth functions and kernels.
    """

    def __init__(
        self,
        growth_function: Statistical_growth_function,
        size: int,
        mus: list | None = None,
        sigmas: list | None = None,
        b: list | None = None,
        kernels: list[dict] | None = None,
        multi_channel: bool = False,
        species_type: Species_types | None = None,
    ) -> None:
        """
        Initialise the filter with given parameters

        Args:
            growth_function (Statistical_growth_function): The growth function to use
            size (int): The size of the filter
            mus (list | None, optional): The means for the growth function. Defaults to None.
            sigmas (list | None, optional): The standard deviations for the growth function. Defaults to None.
            b (list | None, optional): The amplitudes for the growth function. Defaults to None.
            kernels (list[dict] | None, optional): The kernels for multi-kernel convolution. Defaults to None.
            multi_channel (bool, optional): Whether the filter is multi-channel. Defaults to False.
            species_type (Species_types | None, optional): The type of species. Defaults to None.
        """
        self.growth_function = growth_function
        self.size = size
        self.mus = mus
        self.sigmas = sigmas
        self.b = b
        self.kernels = kernels
        self.world_size = (CONST.BOARD_SIZE, CONST.BOARD_SIZE)
        self.R = size // 2
        self.y, self.x = torch.meshgrid(
            torch.arange(-self.R, self.R + 1),
            torch.arange(-self.R, self.R + 1),
            indexing="ij",
        )
        self.distance = torch.sqrt(self.x**2 + self.y**2)
        self.r = self.distance / self.R
        self.multi_channel = multi_channel
        self.species_type = species_type

        if mus is not None and sigmas is not None:
            assert len(mus) == len(sigmas)

        # Préparation des kernels FFT dès l'init pour Fish
        self.prepared_kernels_fft: list[torch.Tensor] | None = None
        if self.kernels is not None:
            self.prepared_kernels_fft = self.prepare_kernels_fft()

    def prepare_kernels_fft(self) -> list:
        """
        Prepare the kernels FFT for multi-kernels convolution
        Returns:
            list: List of kernels in FFT format
        """
        H, W = self.world_size
        mid_h, mid_w = H // 2, W // 2
        kernels_fft = []
        if self.kernels is not None:
            for k in self.kernels:
                # Correction : meshgrid de -mid à mid-1 pour couvrir toute la grille
                y, x = torch.meshgrid(
                    torch.arange(-mid_h, H - mid_h, device="cpu"),
                    torch.arange(-mid_w, W - mid_w, device="cpu"),
                    indexing="ij",
                )
                r_k = max(1e-6, self.R * k.get("r", 1.0))
                D = torch.sqrt(x**2 + y**2) / r_k * len(k["b"])
                amplitudes = torch.tensor(k["b"])[
                    torch.minimum(D.to(torch.int64), torch.tensor(len(k["b"]) - 1))
                ]
                K_local = amplitudes * self.growth_function.target(
                    D % 1, 0.5, 0.15, None
                )
                K_local[D >= len(k["b"])] = 0
                K_local /= K_local.sum() if K_local.sum() > 0 else 1
                kernels_fft.append(torch.fft.fft2(torch.fft.fftshift(K_local)))
            return kernels_fft
        return []

    def filtrer(self):
        """
        Apply the filter to the world state X
        Returns:
            torch.Tensor: The filtered world state in FFT format
        """
        # Fish mode
        if self.kernels is not None:
            return self.prepared_kernels_fft

        # Classic Lenia mode
        H, W = self.world_size
        K_local = torch.zeros_like(self.distance)

        if self.b is not None:
            b_arr = torch.tensor(self.b)
            D = self.distance / self.R * len(b_arr)
            ring_indices = torch.minimum(
                D.to(torch.int64), torch.tensor(len(b_arr) - 1)
            )
            amplitudes = b_arr[ring_indices]
            K_local = amplitudes * self.growth_function.target(D % 1, 0.5, 0.15, None)
            K_local[D >= len(b_arr)] = 0
        elif self.mus is not None and self.sigmas is not None:
            for mu, sigma in zip(self.mus, self.sigmas):
                K_local += self.growth_function.gauss_kernel(self.r, mu, sigma)
            K_local[self.r > 1] = 0
        else:
            raise ValueError("If no b, mus and sigmas must be provided")

        # Normalisation
        K_local /= K_local.sum() if K_local.sum() > 0 else 1

        # Padding to grid format
        K = torch.zeros((H, W), dtype=torch.float)
        kh, kw = K_local.shape
        cy, cx = kh // 2, kw // 2
        K[:kh, :kw] = K_local
        K = torch.roll(K, -cy, dims=0)
        K = torch.roll(K, -cx, dims=1)

        return torch.fft.fft2(K)

    def evolve_lenia(
        self, X: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Evolve the world state X by one time step using the filter
        """
        # --- Fish mode ---
        if (
            self.kernels is not None
            and not self.multi_channel
            and self.species_type == Species_types.FISH
        ):
            # Toujours préparer les kernels FFT si besoin
            if self.prepared_kernels_fft is None:
                self.prepared_kernels_fft = self.prepare_kernels_fft()
            Ks = self.prepared_kernels_fft
            if Ks is None or len(Ks) == 0:
                raise ValueError("Kernels FFT non préparés pour Fish evolution.")
            Us = [torch.real(torch.fft.ifft2(fK * torch.fft.fft2(X))) for fK in Ks]
            Gs = [
                self.growth_function.bell_growth(U, k["m"], k["s"])
                for U, k in zip(Us, self.kernels)
            ]
            X = torch.clamp(X + CONST.DT * torch.mean(torch.stack(Gs), dim=0), 0, 1)  # type: ignore
            return X  # <-- il manquait le return ici dans certains cas

        # --- Multi-channel mode (Aquarium / Emitter / Pacman) ---
        elif (
            self.kernels is not None
            and self.multi_channel
            and self.species_type
            in (Species_types.AQUARIUM, Species_types.EMITTER, Species_types.PACMAN)
        ):
            fXs = [torch.fft.fft2(Xi) for Xi in X]
            Gs = [torch.zeros_like(Xi) for Xi in X]

            if self.species_type == Species_types.AQUARIUM:
                sources, dests, hs, ms, ss = (
                    CONST.SOURCE_AQUARIUM,
                    CONST.DESTINATION_AQUARIUM,
                    CONST.AQUARIUM_hs,
                    CONST.AQUARIUM_ms,
                    CONST.AQUARIUM_ss,
                )
            elif self.species_type == Species_types.EMITTER:
                sources, dests, hs, ms, ss = (
                    CONST.SOURCE_EMITTER,
                    CONST.DESTINATION_EMITTER,
                    CONST.EMITTER_hs,
                    CONST.EMITTER_ms,
                    CONST.EMITTER_ss,
                )
            elif self.species_type == Species_types.PACMAN:
                sources, dests, hs, ms, ss = (
                    CONST.SOURCE_PACMAN,
                    CONST.DESTINATION_PACMAN,
                    CONST.PACMAN_hs,
                    CONST.PACMAN_ms,
                    CONST.PACMAN_ss,
                )
            else:
                raise ValueError("Species type non supporté pour multi-channel.")

            n_channels = len(X)
            funcs = [self.growth_function.bell_growth] * n_channels
            if n_channels == 3 and self.species_type == Species_types.EMITTER:
                funcs[2] = self.growth_function.target  # type: ignore

            if self.prepared_kernels_fft is None:
                raise ValueError("prepared_kernels_fft is None, cannot proceed.")

            for i, (fK, k) in enumerate(zip(self.prepared_kernels_fft, self.kernels)):
                src, dst = sources[i], dests[i]
                h, m, s = hs[i], ms[i], ss[i]
                U = torch.real(torch.fft.ifft2(fK * fXs[src]))
                func = funcs[dst] if dst < len(funcs) else funcs[0]

                if func is funcs[2]:
                    A_dst = X[dst]
                    Gi = func(U, m, s, A_dst)  # type: ignore
                else:
                    Gi = func(U, m, s, None)
                Gs[dst] += h * Gi
            if self.species_type == Species_types.PACMAN:
                return [
                    self.growth_function.soft_clip(Xi + CONST.DT * Gi, 0, 1)
                    for Xi, Gi in zip(X, Gs)
                ]
            else:
                return [torch.clamp(Xi + CONST.DT * Gi, 0, 1) for Xi, Gi in zip(X, Gs)]

        # --- Classic Lenia ---
        else:
            K = self.filtrer()
            if K is None or isinstance(K, list):
                raise ValueError("Kernel invalid for classic Lenia.")

            if isinstance(X, torch.Tensor):
                x = X.clone().detach()
            else:
                raise TypeError("X must be a torch.Tensor for WANDERER species type.")
            U = torch.real(torch.fft.ifft2(torch.fft.fft2(x) * K))

            if self.species_type == Species_types.WANDERER:
                if isinstance(X, torch.Tensor):
                    x = X.clone().detach()
                else:
                    raise TypeError(
                        "X must be a torch.Tensor for WANDERER species type."
                    )
                target = self.growth_function.target(U, CONST.WANDERER_M, CONST.WANDERER_S, None)
                x = torch.clamp(x + CONST.DT * (target - x), 0, 1)
                return x
            else:
                m = CONST.mu if CONST.mu is not None else CONST.GENERIC_M
                s = CONST.sigma if CONST.sigma is not None else CONST.GENERIC_S
                x = torch.clamp(
                    x + CONST.DT * self.growth_function.gaussienne(U, s, m), 0, 1
                )
                return x
