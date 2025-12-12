from board import GameOfLifeBase

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp


class Simulation:
    def __init__(self, size : int) -> None:
        self.size = size
        self.game = GameOfLifeBase(size=self.size)
        self.__run()
    
    def __run(self):
        fig, ax = plt.subplots()
        plt.subplot(111)
        img = ax.imshow(self.game.get_board, cmap='inferno', interpolation='none')
        ax.set_title("Lenia")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
    