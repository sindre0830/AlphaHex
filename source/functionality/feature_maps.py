# external libraries
import numpy as np


def constant_plane(board_size: int, value: float) -> np.ndarray:
    return np.full(shape=(board_size, board_size), fill_value=value, dtype=np.float32)
