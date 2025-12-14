import numpy as np


class Board:
    def __init__(self, size: int, void: bool) -> None:
        self.__size = size
        self.__board: np.ndarray = np.random.rand(size, size) if not void else np.zeros((size, size))

    @property
    def get_size(self) -> int:
        return self.__size

    @property
    def get_board(self) -> np.ndarray:
        return self.__board
