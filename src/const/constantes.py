from .pattern import pattern
from species.species_types import Species_types
from typing import Any

# -----------BOARD---------------------
BOARD_SIZE: int = 100
VOID_BOARD: bool = True

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
AQUARIUM_CELLS = pattern["aquarium"].get("cells", [])

# ----------- PARAMETRES GENERIQUES -----------
GENERIC_M: float = 0.5  # mu générique pour formes arbitraires
GENERIC_S: float = 0.1  # sigma générique
GENERIC_T: int = 10
GENERIC_R: int = 20

# --- CHOIX ACTIF (orbium, hydrogeminium, ou generic) ---
USE_ORBIUM_PARAMS: bool = False  # si True: utilise ORBIUM_M/S/T/R
USE_HYDROGEMINIUM_PARAMS: bool = False  # si True: utilise HYDROGEMINIUM_M/S/T/R
USE_FISH_PARAMS: bool = True
USE_AQUARIUM_PARAMS: bool = False
# Si les deux sont False: utilise GENERIC_M/S/T/R

# -----------CROISSANCE----------------
# Sélectionner paramètres selon le mode actif
if USE_ORBIUM_PARAMS:
    SIGMA: float = ORBIUM_S
    MU: float = ORBIUM_M
    ACTIVE_R: int = ORBIUM_R
    ACTIVE_T: float = ORBIUM_T
    KERNEL_TYPE = Species_types.ORBIUM
elif USE_HYDROGEMINIUM_PARAMS:
    SIGMA: float = HYDROGEMINIUM_S
    MU: float = HYDROGEMINIUM_M
    ACTIVE_R: int = HYDROGEMINIUM_R
    ACTIVE_T: float = HYDROGEMINIUM_T
    KERNEL_TYPE = Species_types.HYDROGEMINIUM
elif USE_FISH_PARAMS:
    SIGMA: float | None = None
    MU: float | None = None
    ACTIVE_R: int = FISH_R
    ACTIVE_T: float = FISH_T
    KERNEL_TYPE = Species_types.FISH
elif USE_AQUARIUM_PARAMS:
    SIGMA: float | None = None
    MU: float | None = None
    ACTIVE_R: int = AQUARIUM_R
    ACTIVE_T: float = AQUARIUM_T
    KERNEL_TYPE = Species_types.AQUARIUM
else:
    SIGMA: float = GENERIC_S
    MU: float = GENERIC_M
    ACTIVE_R: int = GENERIC_R
    ACTIVE_T: float = GENERIC_T
    KERNEL_TYPE = Species_types.GENERIC


DT: float = 1.0 / ACTIVE_T

# -----------FILTRE--------------------
FILTRE_SIZE: int = 2 * ACTIVE_R + 1

# ------------KERNEL----------------
# Mus est la liste des positions des pics du kernel
# Sigmas est la liste des largeurs des pics du kernel

# Kernel à une coquille
# MUS = [0.3]
# SIGMAS = [0.05]

# Kernel à deux coquilles
# MUS = [0.3, 0.7]
# SIGMAS = [0.05, 0.1]

# -------------Kernel du code référence pour generic-----------------
MUS = [0.5]
SIGMAS = [0.15]

# Kernel nerveux
# MUS = [0.2]
# SIGMAS = [0.03]
