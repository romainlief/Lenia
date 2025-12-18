import numpy as np
from const.constantes import BOARD_SIZE, VOID_BOARD, CHANNEL_COUNT


class Board:
    def __init__(
        self,
        size: int = BOARD_SIZE,
        channels: int = CHANNEL_COUNT,
        void: bool = VOID_BOARD,
    ) -> None:
        self.__size = size
        self.__channels = channels

        if void:
            self.__board = np.zeros((size, size, channels), dtype=np.float32)
        else:
            self.__board = np.random.rand(size, size, channels).astype(np.float32)

    @property
    def size(self) -> int:
        return self.__size

    @property
    def channels(self) -> int:
        return self.__channels

    @property
    def board(self) -> np.ndarray:
        return self.__board
