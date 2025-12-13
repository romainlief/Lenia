from simulation import Simulation
from constantes import *
from type_croissance import Type_de_croissance
from croissance import Fonction_de_croissance

if __name__ == "__main__":
    sim = Simulation(size=BOARD_SIZE)
    # Placer orbium avec ses paramètres spécifiques
    sim.seed_orbium(ORBIUM, center=(BOARD_SIZE//2, BOARD_SIZE//2), rotate=0, amplitude=1.0, normalize=True)
    sim.run()
    