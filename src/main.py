from simulation import Simulation
from constantes import *
from type_croissance import Type_de_croissance
from croissance import Fonction_de_croissance

if __name__ == "__main__":
    try:
        sim = Simulation(size=BOARD_SIZE)
        sim.seed_orbium(pattern["orbium"].get("cells"), center=(BOARD_SIZE//2, BOARD_SIZE//2), rotate=0, amplitude=1.0, normalize=True)
        sim.run()
    except KeyboardInterrupt:
        print("Simulation interrompue par l'utilisateur.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
    