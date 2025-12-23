from const.constantes import SIMULATION_MODE
from launcher.simulation.simulation import Simulation
from launcher.simulation.GUILogic import GUILogic
class Launcher:
    def __init__(self, mode=SIMULATION_MODE) -> None:
        self.mode = mode
    
    def launch(self) -> None:
        if self.mode:
            sim_launcher = GUILogic(Simulation())
            sim_launcher.run()
        else:
            pass  # Placeholder for Exploration mode