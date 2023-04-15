# internal libraries
from constants import (
    BRIDGE_DIRECTIONS,
    EMPTY
)
from functionality.board import (
    in_bounds,
    cells_between,
    get_distance_from_center,
    illegal_actions,
    in_bound_neighbours
)
from functionality.game import (
    opposite_player
)
# external libraries
import numpy as np


def bridge_templates(board: np.ndarray, player: int) -> np.ndarray:
    board_size = len(board)
    bridge_actions = np.zeros_like(board, dtype=np.float32)
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] == player:
                # iterates through each neighbour cell which would result in a bridge
                for direction_row, direction_col in BRIDGE_DIRECTIONS:
                    bridge_row = row + direction_row
                    bridge_col = col + direction_col
                    if not in_bounds(board_size, cell=(bridge_row, bridge_col)):
                        continue
                    # branch if neighbour cell is occupied
                    if board[bridge_row][bridge_col] != EMPTY:
                        continue
                    # make sure none of the cells between are occupied
                    cells_between_occupied = False
                    for cell_between_row, cells_between_col in cells_between(cell=(row, col), target=(bridge_row, bridge_col)):
                        if board[cell_between_row][cells_between_col] != EMPTY:
                            cells_between_occupied = True
                            break
                    if cells_between_occupied:
                        continue
                    # add 1 so that bridge actions that overlap are differentiated
                    bridge_actions[bridge_row][bridge_col] += 1
    return bridge_actions


def critical_bridge_connections(board: np.ndarray, player: int) -> np.ndarray:
    board_size = len(board)
    opponent = opposite_player(player)
    feature = np.zeros_like(board, dtype=np.float32)
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] == player:
                # iterates through each neighbour cell which would result in a bridge
                for direction_row, direction_col in BRIDGE_DIRECTIONS:
                    bridge_row = row + direction_row
                    bridge_col = col + direction_col
                    if not in_bounds(board_size, cell=(bridge_row, bridge_col)):
                        continue
                    # branch if neighbour cell isn't occupied by player
                    if board[bridge_row][bridge_col] != player:
                        continue
                    # make sure none of the cells between are occupied
                    chain_connected = False
                    critical = False
                    critical_cells = []
                    for cell_between_row, cells_between_col in cells_between(cell=(row, col), target=(bridge_row, bridge_col)):
                        if board[cell_between_row][cells_between_col] == player:
                            chain_connected = True
                            break
                        elif board[cell_between_row][cells_between_col] != opponent:
                            critical = True
                            critical_cells.append((cell_between_row, cells_between_col))
                    if chain_connected or not critical:
                        continue
                    for critical_cell_row, critical_cell_col in critical_cells:
                        feature[critical_cell_row][critical_cell_col] = 1
    return feature


def block(board: np.ndarray, player: int) -> np.ndarray:
    board_size = len(board)
    opponent = opposite_player(player)
    feature = np.zeros_like(board, dtype=np.float32)
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] == opponent:
                for neighbour_row, neighbour_col in in_bound_neighbours(board_size, cell=(row, col)):
                    if board[neighbour_row][neighbour_col] == EMPTY:
                        feature[neighbour_row][neighbour_col] += 1
    return feature


def center_importance(board: np.ndarray, player: int) -> np.ndarray:
    board_size = len(board)
    distance_from_center = get_distance_from_center(board_size)
    for row, col in illegal_actions(board):
        distance_from_center[row][col] = 0
    return distance_from_center
