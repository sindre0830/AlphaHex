# internal libraries
from state_manager import StateManager
# external libraries
import numpy as np


def constant_plane(board: np.ndarray, value: float) -> np.ndarray:
    return np.full_like(board, fill_value=value, dtype=np.float32)


def onehot_encode_cell(board: np.ndarray, target: int) -> np.ndarray:
    return np.where(board == target, 1, 0).astype(dtype=np.float32)


def sensibleness(board: np.ndarray) -> np.ndarray:
    feature = np.zeros_like(board, dtype=np.float32)
    state = StateManager()
    state.initialize_state(grid_size=len(board))
    for (row, column) in state.legal_actions(board):
        feature[row][column] = 1
    return feature


def strategy(strategy_func, board: np.ndarray, player: int) -> np.ndarray:
    return strategy_func(board, player)
