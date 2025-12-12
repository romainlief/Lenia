import numpy as np


class GameOfLifeBase:
    def __init__(self, size: int) -> None:
        self.__size = size
        self.__board: np.ndarray = np.random.rand(size, size)

    @property
    def get_size(self) -> int:
        return self.__size

    @property
    def get_board(self) -> np.ndarray:
        return self.__board
