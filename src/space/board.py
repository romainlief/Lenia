import numpy as np
from const.constantes import BOARD_SIZE, VOID_BOARD, CHANNEL_COUNT


class Board:
    """
    Class representing the simulation board.
    """

    def __init__(
        self,
        size: int = BOARD_SIZE,
        channels: int = CHANNEL_COUNT,
        void: bool = VOID_BOARD,
    ) -> None:
        """
            Initializes the board.

        Args:
            size (int, optional): The size of the board. Defaults to BOARD_SIZE.
            channels (int, optional): The number of channels. Defaults to CHANNEL_COUNT.
            void (bool, optional): Whether the board is void (empty). Defaults to VOID_BOARD.
        """
        self.__size = size
        self.__channels = channels

        if void:
            self.__board = np.zeros((size, size, channels), dtype=np.float32)
        else:
            self.__board = np.random.rand(size, size, channels).astype(np.float32)

    @property
    def size(self) -> int:
        """
        Get the size of the board.

        Returns:
            int: The size of the board.
        """
        return self.__size

    @property
    def channels(self) -> int:
        """
        Get the number of channels.

        Returns:
            int: The number of channels.
        """
        return self.__channels

    @property
    def board(self) -> np.ndarray:
        """
        Get the board array.

        Returns:
            np.ndarray: The board array.
        """
        return self.__board
