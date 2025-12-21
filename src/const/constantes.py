from .pattern import pattern
from species.species_types import Species_types
import numpy as np

# -----------BOARD---------------------
BOARD_SIZE: int = 100
VOID_BOARD: bool = True
CHANNEL_COUNT: int = 1

# --- CHOOSE ---
USE_ORBIUM_PARAMS: bool = False
USE_HYDROGEMINIUM_PARAMS: bool = False
USE_FISH_PARAMS: bool = False
USE_AQUARIUM_PARAMS: bool = False
USE_WANDERER_PARAMS: bool = False
USE_EMITTER_PARAMS: bool = False
USE_PACMAN_PARAMS: bool = True

# ----------- PARAMETRES ORBIUM -----------
ORBIUM_M: float = pattern["orbium"].get("m", 0)  # mu de la croissance pour cet orbium
ORBIUM_S: float = pattern["orbium"].get(
    "s", 0
)  # sigma de la croissance pour cet orbium
ORBIUM_T: float = pattern["orbium"].get("T", 0)  # pas temporel inverse (DT = 1/T)
ORBIUM_R: int = pattern["orbium"].get("R", 0)  # rayon du noyau pour cet orbium
ORBIUM_CELLS = pattern["orbium"].get("cells", [])
ORBIUM_B: list = pattern["orbium"].get("b", [])

# ----------- PARAMETRES HYDROGEMINIUM -----------
HYDROGEMINIUM_M: float = pattern["geminium"].get(
    "m", 0
)  # mu de la croissance pour hydrogeminium
HYDROGEMINIUM_S: float = pattern["geminium"].get(
    "s", 0
)  # sigma de la croissance pour hydrogeminium (AFFINE: 0.036 -> 0.022)
HYDROGEMINIUM_T: float = pattern["geminium"].get(
    "T", 0
)  # pas temporel inverse (ralentit l'évolution)
HYDROGEMINIUM_R: int = pattern["geminium"].get(
    "R", 0
)  # rayon du noyau (max(b) = 3 niveaux)
HYDROGEMINIUM_CELLS = pattern["geminium"].get("cells", [])
HYDROGEMINIUM_B: list = pattern["geminium"].get("b", [])

# ----------- PARAMETRES FISH -----------
FISH_R: int = pattern["fish"].get("R", 0)
FISH_T: int = pattern["fish"].get("T", 0)
FISH_KERNEL = pattern["fish"].get("kernels", [])
FISH_CELLS = pattern["fish"].get("cells", [])

# ----------- PARAMETRES AQUARIUM -----------
AQUARIUM_R: int = pattern["aquarium"].get("R", 0)
AQUARIUM_T: int = pattern["aquarium"].get("T", 0)
AQUARIUM_KERNEL = pattern["aquarium"].get("kernels", [])
DESTINATION_AQUARIUM = [k["c1"] for k in AQUARIUM_KERNEL]
SOURCE_AQUARIUM = [k["c0"] for k in AQUARIUM_KERNEL]
AQUARIUM_H = [k["h"] for k in AQUARIUM_KERNEL]
AQUARIUM_bs = [k["b"] for k in AQUARIUM_KERNEL]
AQUARIUM_rs = [AQUARIUM_R * k["r"] for k in AQUARIUM_KERNEL]
AQUARIUM_ms = [k["m"] for k in AQUARIUM_KERNEL]
AQUARIUM_ss = [k["s"] for k in AQUARIUM_KERNEL]
AQUARIUM_hs = [k["h"] for k in AQUARIUM_KERNEL]
AQUARIUM_CELLS = np.array(pattern["aquarium"]["cells"], dtype=np.float32)

# ----------- PARAMETRES WANDERER -----------
WANDERER_M: float = pattern["wanderer"].get("m", 0)
WANDERER_S: float = pattern["wanderer"].get("s", 0)
WANDERER_T: float = pattern["wanderer"].get("T", 0)
WANDERER_R: int = pattern["wanderer"].get("R", 0)
WANDERER_CELLS = pattern["wanderer"].get("cells", [])
WANDERER_B: list = pattern["wanderer"].get("b", [])

# ----------- PARAMETRES EMITTER -----------
EMITTER_R: int = pattern["emitter"].get("R", 0)
EMITTER_T: int = pattern["emitter"].get("T", 0)
EMITTER_KERNEL = pattern["emitter"].get("kernels", [])
DESTINATION_EMITTER = [k["c1"] for k in EMITTER_KERNEL]
SOURCE_EMITTER = [k["c0"] for k in EMITTER_KERNEL]
EMITTER_H = [k["h"] for k in EMITTER_KERNEL]
EMITTER_bs = [k["b"] for k in EMITTER_KERNEL]
EMITTER_rs = [EMITTER_R * k["r"] for k in EMITTER_KERNEL]
EMITTER_ms = [k["m"] for k in EMITTER_KERNEL]
EMITTER_ss = [k["s"] for k in EMITTER_KERNEL]
EMITTER_hs = [k["h"] for k in EMITTER_KERNEL]
EMITTER_CELLS = np.array(pattern["emitter"]["cells"], dtype=np.float32)

# ----------- PARAMETRES PACMAN -----------
PACMAN_R: int = pattern["pacman"].get("R", 0)
PACMAN_T: int = pattern["pacman"].get("T", 0)
PACMAN_KERNEL = pattern["pacman"].get("kernels", [])
DESTINATION_PACMAN = [k["c1"] for k in PACMAN_KERNEL]
SOURCE_PACMAN = [k["c0"] for k in PACMAN_KERNEL]
PACMAN_H = [k["h"] for k in PACMAN_KERNEL]
PACMAN_bs = [k["b"] for k in PACMAN_KERNEL]
PACMAN_rs = [PACMAN_R * k["r"] for k in PACMAN_KERNEL]
PACMAN_ms = [k["m"] for k in PACMAN_KERNEL]
PACMAN_ss = [k["s"] for k in PACMAN_KERNEL]
PACMAN_hs = [k["h"] for k in PACMAN_KERNEL]
PACMAN_CELLS = np.array(pattern["pacman"]["cells"], dtype=np.float32)

# ----------- PARAMETRES GENERIQUES -----------
GENERIC_M: float = 0.5  # mu générique pour formes arbitraires
GENERIC_S: float = 0.1  # sigma générique
GENERIC_T: int = 10
GENERIC_R: int = 20

# -----------PARAMETERS----------------
if USE_ORBIUM_PARAMS:
    SIGMA: float = ORBIUM_S
    MU: float = ORBIUM_M
    ACTIVE_R: int = ORBIUM_R
    ACTIVE_T: float = ORBIUM_T
    KERNEL_TYPE = Species_types.ORBIUM
    CHANNEL_COUNT = 1
elif USE_HYDROGEMINIUM_PARAMS:
    SIGMA: float = HYDROGEMINIUM_S
    MU: float = HYDROGEMINIUM_M
    ACTIVE_R: int = HYDROGEMINIUM_R
    ACTIVE_T: float = HYDROGEMINIUM_T
    KERNEL_TYPE = Species_types.HYDROGEMINIUM
    CHANNEL_COUNT = 1
elif USE_FISH_PARAMS:
    SIGMA: float | None = None
    MU: float | None = None
    ACTIVE_R: int = FISH_R
    ACTIVE_T: float = FISH_T
    KERNEL_TYPE = Species_types.FISH
    CHANNEL_COUNT = 1
elif USE_AQUARIUM_PARAMS:
    SIGMA: float | None = None
    MU: float | None = None
    ACTIVE_R: int = AQUARIUM_R
    ACTIVE_T: float = AQUARIUM_T
    KERNEL_TYPE = Species_types.AQUARIUM
    CHANNEL_COUNT = 3
elif USE_EMITTER_PARAMS:
    SIGMA: float | None = None
    MU: float | None = None
    ACTIVE_R: int = EMITTER_R
    ACTIVE_T: float = EMITTER_T
    KERNEL_TYPE = Species_types.EMITTER
    CHANNEL_COUNT = 3
elif USE_WANDERER_PARAMS:
    SIGMA: float = WANDERER_S
    MU: float = WANDERER_M
    ACTIVE_R: int = WANDERER_R
    ACTIVE_T: float = WANDERER_T
    KERNEL_TYPE = Species_types.WANDERER
    CHANNEL_COUNT = 1
elif USE_PACMAN_PARAMS:
    SIGMA: float | None = None
    MU: float | None = None
    ACTIVE_R: int = PACMAN_R
    ACTIVE_T: float = PACMAN_T
    KERNEL_TYPE = Species_types.PACMAN
    CHANNEL_COUNT = 3
else:
    SIGMA: float = GENERIC_S
    MU: float = GENERIC_M
    ACTIVE_R: int = GENERIC_R
    ACTIVE_T: float = GENERIC_T
    KERNEL_TYPE = Species_types.GENERIC
    CHANNEL_COUNT = 1

# -----------TIME STEP-----------------
DT: float = 1.0 / ACTIVE_T

# -----------FILTRE--------------------
FILTRE_SIZE: int = 2 * ACTIVE_R + 1

# -------------Kernel du code référence pour generic-----------------
MUS = [0.5]
SIGMAS = [0.15]
