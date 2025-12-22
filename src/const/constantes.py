from .pattern import pattern
from species.species_types import Species_types
import torch

# -----------BOARD---------------------
BOARD_SIZE: int = 100
VOID_BOARD: bool = True
channel_count: int = 1
SIMULATION_MODE: bool = True  # True for simulation, False for Exploration

# --- CHOOSE ---
USE_ORBIUM_PARAMS: bool = True
USE_HYDROGEMINIUM_PARAMS: bool = False
USE_FISH_PARAMS: bool = False
USE_AQUARIUM_PARAMS: bool = False
USE_WANDERER_PARAMS: bool = False
USE_EMITTER_PARAMS: bool = False  
USE_PACMAN_PARAMS: bool = False 

# ----------- PARAMETRES ORBIUM -----------
ORBIUM_M: float = pattern["orbium"].get("m", 0)  # mu de la croissance pour cet orbium
ORBIUM_S: float = pattern["orbium"].get(
    "s", 0
)  # sigma de la croissance pour cet orbium
ORBIUM_T: float = pattern["orbium"].get("T", 0)  # pas temporel inverse (DT = 1/T)
ORBIUM_R: int = pattern["orbium"].get("R", 0)  # rayon du noyau pour cet orbium
ORBIUM_CELLS = pattern["orbium"].get("cells", [])
ORBIUM_B: list[float] = pattern["orbium"].get("b", [])

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
HYDROGEMINIUM_B: list[float] = pattern["geminium"].get("b", [])

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
AQUARIUM_CELLS = torch.tensor(pattern["aquarium"]["cells"], dtype=torch.float32)

# ----------- PARAMETRES WANDERER -----------
WANDERER_M: float = pattern["wanderer"].get("m", 0)
WANDERER_S: float = pattern["wanderer"].get("s", 0)
WANDERER_T: float = pattern["wanderer"].get("T", 0)
WANDERER_R: int = pattern["wanderer"].get("R", 0)
WANDERER_CELLS = pattern["wanderer"].get("cells", [])
WANDERER_B: list[float] = pattern["wanderer"].get("b", [])

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
EMITTER_CELLS = torch.tensor(pattern["emitter"]["cells"], dtype=torch.float32)

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
PACMAN_CELLS = torch.tensor(pattern["pacman"]["cells"], dtype=torch.float32)

# ----------- PARAMETRES GENERIQUES -----------
GENERIC_M: float = 0.5  # mu générique pour formes arbitraires
GENERIC_S: float = 0.1  # sigma générique
GENERIC_T: int = 10
GENERIC_R: int = 20

# -----------PARAMETERS----------------
if USE_ORBIUM_PARAMS:
    sigma = ORBIUM_S
    mu = ORBIUM_M
    active_r = ORBIUM_R
    active_t = ORBIUM_T
    kernel_type = Species_types.ORBIUM
    channel_count = 1
elif USE_HYDROGEMINIUM_PARAMS:
    sigma = HYDROGEMINIUM_S
    mu = HYDROGEMINIUM_M
    active_r = HYDROGEMINIUM_R
    active_t = HYDROGEMINIUM_T
    kernel_type = Species_types.HYDROGEMINIUM
    channel_count = 1
elif USE_FISH_PARAMS:
    sigma = None  # type: float | None
    mu = None
    active_r = FISH_R
    active_t = FISH_T
    kernel_type = Species_types.FISH
    channel_count = 1
elif USE_AQUARIUM_PARAMS:
    sigma = None
    mu = None
    active_r = AQUARIUM_R
    active_t = AQUARIUM_T
    kernel_type = Species_types.AQUARIUM
    channel_count = 3
elif USE_EMITTER_PARAMS:
    sigma = None
    mu = None
    active_r = EMITTER_R
    active_t = EMITTER_T
    kernel_type = Species_types.EMITTER
    channel_count = 3
elif USE_WANDERER_PARAMS:
    sigma = WANDERER_S
    mu = WANDERER_M
    active_r = WANDERER_R
    active_t = WANDERER_T
    kernel_type = Species_types.WANDERER
    channel_count = 1
elif USE_PACMAN_PARAMS:
    sigma = None
    mu = None
    active_r = PACMAN_R
    active_t = PACMAN_T
    kernel_type = Species_types.PACMAN
    channel_count = 3
else:
    sigma = GENERIC_S
    mu = GENERIC_M
    active_r = GENERIC_R
    active_t = GENERIC_T
    kernel_type = Species_types.GENERIC
    channel_count = 1

# -----------TIME STEP-----------------
DT: float = 1.0 / active_t

# -----------FILTRE--------------------
FILTRE_SIZE: int = 2 * active_r + 1

# -------------Kernel du code référence pour generic-----------------
MUS = [0.5]
SIGMAS = [0.15]
