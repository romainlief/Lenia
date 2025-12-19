from launcher.simulation import Simulation
from const.constantes import (
    BOARD_SIZE,
    KERNEL_TYPE,
    USE_HYDROGEMINIUM_PARAMS,
    USE_ORBIUM_PARAMS,
    USE_FISH_PARAMS,
    USE_WANDERER_PARAMS,
    USE_AQUARIUM_PARAMS,
    USE_EMITTER_PARAMS,
    USE_PACMAN_PARAMS
)
from species.orbium import Orbium
from species.hydrogeminium import Hydrogeminium
from species.fish import Fish
from species.wanderer import Wanderer


if __name__ == "__main__":
    sim = Simulation(kernel_type=KERNEL_TYPE)
    if USE_ORBIUM_PARAMS:
        orbium = Orbium()
        patch = orbium.make_patch(
            rotate=0,
            amplitude=4.0,
            normalize=True,
        )
    elif USE_HYDROGEMINIUM_PARAMS:
        hydrogenium = Hydrogeminium()
        patch = hydrogenium.make_patch(
            rotate=0,
            amplitude=4.0,
            normalize=True,
        )
    elif USE_FISH_PARAMS:
        fish = Fish()
        patch = fish.make_patch(
            rotate=0,
            amplitude=4.0,
            normalize=True,
        )
    elif USE_WANDERER_PARAMS:
        wanderer = Wanderer()
        patch = wanderer.make_patch(
            rotate=0,
            amplitude=4.0,
            normalize=True,
        )
    if not USE_AQUARIUM_PARAMS and not USE_EMITTER_PARAMS and not USE_PACMAN_PARAMS:
        sim.apply_patch(patch, center=(BOARD_SIZE // 2, BOARD_SIZE // 2))
    sim.run()
