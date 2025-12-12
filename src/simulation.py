from board import GameOfLifeBase
from croissance import Fonction_de_croissance
from filtre import Filtre
from constantes import *
from type_croissance import Type_de_croissance

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp


class Simulation:
    def __init__(self, size: int) -> None:
        self.size = size
        self.game = GameOfLifeBase(size=self.size)
        self.filtre = Filtre(
            fonction_de_croissance=Fonction_de_croissance(
                type=Type_de_croissance.GAUSSIENNE
            ),
            size=FILTRE_SIZE,
            mus=[MU],
            sigmas=[SIGMA],
        )
        self.X = self.game.get_board
        self.__run()

    def __update(self, frame: int) -> list:
        # évolution du board
        self.X = self.filtre.evolve_lenia(self.X)
        self.img.set_data(self.X)
        return [self.img]

    def __run(self):
        fig, ax = plt.subplots()
        self.img = ax.imshow(self.X, cmap="inferno", interpolation="none")
        ax.set_title("Lenia")
        ax.set_xticks([])
        ax.set_yticks([])
        anim = animation.FuncAnimation(
            fig, self.__update, frames=200, interval=20, blit=True
        )
        plt.show()
