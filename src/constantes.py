import numpy as np
# ------------BOARD---------------------
BOARD_SIZE: int = 100

# ------------CROISSANCE----------------
U: np.ndarray = np.arange(0, 0.3, 0.001)
SIGMA: float = 0.15
MU: float = 0.5
DT : float = 0.1

# ------------FILTRE--------------------
MUS : list = [0.3, 0.7]
SIGMAS : list = [0.1, 0.2]
FILTRE_SIZE : int = 25