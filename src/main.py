from launcher.simulation import Simulation
from const.constantes import *
from species.species_types import Species_types
from species.orbium import Orbium
from species.hydrogeminium import Hydrogeminium

if __name__ == "__main__":
    try:
        sim = Simulation(kernel_type=KERNEL_TYPE)
       
        orbium = Orbium()
        hydrogenium = Hydrogeminium()
        patch = orbium.make_patch(
            rotate=0,
            amplitude=4.0,
            normalize=True,
        )

        sim.apply_patch(patch, center=(BOARD_SIZE // 2, BOARD_SIZE // 2))

        sim.run()
    except KeyboardInterrupt:
        print("Simulation interrompue par l'utilisateur.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
