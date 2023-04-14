# external libraries
import numpy as np


def fork(board: list[list[int]], player: int) -> np.ndarray:
    board_size = len(board)
    two_bridge_actions_board = np.zeros((board_size, board_size), dtype=np.float32)
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] == player:
                # iterates through each neighbour cell which would result in a bridge, order:
                # left, right, top left, top right, bottom left, bottom right
                for direction_row, direction_col in [(-1, -1), (1, 1), (1, -2), (2, -1), (-2, 1), (-1, 2)]:
                    neighbour_row = row + direction_row
                    neighbour_col = col + direction_col
                    # branch if neighbour cell is out of bounds
                    if neighbour_row < 0 or neighbour_row > board_size - 1:
                        continue
                    if neighbour_col < 0 or neighbour_col > board_size - 1:
                        continue
                    # branch if neighbour cell is occupied
                    if board[neighbour_row][neighbour_col] != 0:
                        continue
                    two_bridge_actions_board[neighbour_row][neighbour_col] += 1
    return two_bridge_actions_board


def center_control(board_size: int) -> np.ndarray:
    board = np.zeros((board_size, board_size), dtype=np.float32)
    center = board_size // 2
    size = board_size - 1
    for fill_value in (range(0, center + 1)):
        size -= 1
        board[center - size // 2: center + size // 2 + 1, center - size // 2: center + size // 2 + 1] = fill_value
    return board

