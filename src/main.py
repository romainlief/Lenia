from launcher.simulation import Simulation
from const.constantes import *
from species.species import Species


if __name__ == "__main__":
    try:
        sim = Simulation(size=BOARD_SIZE, kernel_type=Species.HYDROGEMINIUM)
        sim.seed_hydrogeminium(
            pattern["geminium"].get("cells"),
            center=(BOARD_SIZE // 2, BOARD_SIZE // 2),
            rotate=0,
            amplitude=1.0,
            normalize=True,
        )
        #sim.seed_orbium(
        #    pattern["orbium"].get("cells"),
        #    center=(BOARD_SIZE // 4, BOARD_SIZE // 4),
        #    rotate=0,
        #    amplitude=1.0,
        #    normalize=True,
       # )
        sim.run()
    except KeyboardInterrupt:
        print("Simulation interrompue par l'utilisateur.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
