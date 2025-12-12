import numpy as np
# ------------BOARD---------------------
BOARD_SIZE: int = 100

# ------------CROISSANCE----------------
U: np.ndarray = np.arange(0, 0.3, 0.001)
SIGMA: float = 0.15
MU: float = 0.5
DT : float = 0.05

# ------------FILTRE--------------------
FILTRE_SIZE : int = 45

# ------------KERNEL----------------
# Mus est la liste des positions des pics du kernel
# Sigmas est la liste des largeurs des pics du kernel

# Kernel à une coquille
#MUS = [0.3]
#SIGMAS = [0.05]

# Kernel à deux coquilles
#MUS = [0.3, 0.7]
#SIGMAS = [0.05, 0.1]

# Kernel très fluide 
MUS = [0.005] 
SIGMAS = [0.08]

# Kernel nerveux 
#MUS = [0.2]
#SIGMAS = [0.03]
#------------------------------------