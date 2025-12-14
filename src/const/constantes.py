import numpy as np
from typing import Dict, Any

# -----------BOARD---------------------
BOARD_SIZE: int = 100
VOID_BOARD: bool = True

# ----------- PARAMETRES ORBIUM (à définir EN PREMIER) -----------
ORBIUM_M: float = 0.15  # mu de la croissance pour cet orbium
ORBIUM_S: float = 0.015  # sigma de la croissance pour cet orbium
ORBIUM_T: float = 10  # pas temporel inverse (DT = 1/T)
ORBIUM_R: int = 13  # rayon du noyau pour cet orbium

# ----------- PARAMETRES HYDROGEMINIUM (basés sur code référence) -----------
HYDROGEMINIUM_M: float = 0.26  # mu de la croissance pour hydrogeminium
HYDROGEMINIUM_S: float = 0.036  # sigma de la croissance pour hydrogeminium (AFFINE: 0.036 -> 0.022)
HYDROGEMINIUM_T: float = 10  # pas temporel inverse (ralentit l'évolution)
HYDROGEMINIUM_R: int = 18  # rayon du noyau (max(b) = 3 niveaux)

# ----------- PARAMETRES GENERIQUES (pour glider, random orbions, etc.) -----------
GENERIC_M: float = 0.5  # mu générique pour formes arbitraires
GENERIC_S: float = 0.1  # sigma générique
GENERIC_T: int = 10
GENERIC_R: int = 20

# --- CHOIX ACTIF (orbium, hydrogeminium, ou generic) ---
USE_ORBIUM_PARAMS: bool = False  # si True: utilise ORBIUM_M/S/T/R
USE_HYDROGEMINIUM_PARAMS: bool = True  # si True: utilise HYDROGEMINIUM_M/S/T/R
# Si les deux sont False: utilise GENERIC_M/S/T/R

# -----------CROISSANCE----------------
U: np.ndarray = np.arange(0, 1.0, 0.001)

# Sélectionner paramètres selon le mode actif
if USE_ORBIUM_PARAMS:
    SIGMA: float = ORBIUM_S
    MU: float = ORBIUM_M
    ACTIVE_R: int = ORBIUM_R
    ACTIVE_T: float = ORBIUM_T
elif USE_HYDROGEMINIUM_PARAMS:
    SIGMA: float = HYDROGEMINIUM_S
    MU: float = HYDROGEMINIUM_M
    ACTIVE_R: int = HYDROGEMINIUM_R
    ACTIVE_T: float = HYDROGEMINIUM_T
else:
    SIGMA: float = GENERIC_S
    MU: float = GENERIC_M
    ACTIVE_R: int = GENERIC_R
    ACTIVE_T: float = GENERIC_T


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

