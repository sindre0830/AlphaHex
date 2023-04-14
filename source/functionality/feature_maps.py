# internal libraries
from functionality.board import (
    legal_actions
)
# external libraries
import numpy as np


def constant_plane(board_size: int, value: float) -> np.ndarray:
    return np.full(shape=(board_size, board_size), fill_value=value, dtype=np.float32)


def onehot_encode_cell(board: list[list[int]], target: int) -> np.ndarray:
    board_size = len(board)
    feature = np.zeros(shape=(board_size, board_size), dtype=np.float32)
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] == target:
                feature[row][col] = 1
    return feature


def sensibleness(board: list[list[int]]) -> np.ndarray:
    board_size = len(board)
    feature = np.zeros(shape=(board_size, board_size), dtype=np.float32)
    for (row, column) in legal_actions(board):
        feature[row][column] = 1
    return feature


def strategy(strategy_func, board: list[list[int]], player: int) -> np.ndarray:
    return strategy_func(board, player)
