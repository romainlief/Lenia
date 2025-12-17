from launcher.simulation import Simulation
from const.constantes import *
from species.species_types import Species_types
from species.orbium import Orbium
from species.hydrogeminium import Hydrogeminium
from species.fish import Fish
from species.aquarium import Aquarium


if __name__ == "__main__":
    sim = Simulation(kernel_type=KERNEL_TYPE)

    orbium = Orbium()
    hydrogenium = Hydrogeminium()
    fish = Fish()
    aquarium = Aquarium()
    patch = hydrogenium.make_patch(
        rotate=0,
        amplitude=4.0,
        normalize=True,
    )
    sim.apply_patch(patch, center=(BOARD_SIZE // 2, BOARD_SIZE // 2))

    sim.run()
