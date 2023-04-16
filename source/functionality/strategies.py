# internal libraries
from constants import (
    BRIDGE_DIRECTIONS,
    EMPTY,
    PLAYER_1
)
import functionality.hexagon
# external libraries
import numpy as np


def bridge_templates(board: np.ndarray, player: int, opponent: int) -> np.ndarray:
    board_size = len(board)
    bridge_actions = np.zeros_like(board, dtype=np.float32)
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] == player:
                # iterates through each neighbour cell which would result in a bridge
                for direction_row, direction_col in BRIDGE_DIRECTIONS:
                    bridge_row = row + direction_row
                    bridge_col = col + direction_col
                    if not functionality.hexagon.in_bounds(board_size, cell=(bridge_row, bridge_col)):
                        continue
                    # branch if neighbour cell is occupied
                    if board[bridge_row][bridge_col] != EMPTY:
                        continue
                    # make sure none of the cells between are occupied
                    cells_between_occupied = False
                    cells_between = functionality.hexagon.cells_between(
                        cell=(row, col),
                        target=(bridge_row, bridge_col)
                    )
                    for cell_between_row, cells_between_col in cells_between:
                        if board[cell_between_row][cells_between_col] != EMPTY:
                            cells_between_occupied = True
                            break
                    if cells_between_occupied:
                        continue
                    # add 1 so that bridge actions that overlap are differentiated
                    bridge_actions[bridge_row][bridge_col] += 1
    return bridge_actions


def critical_bridge_connections(board: np.ndarray, player: int, opponent: int) -> np.ndarray:
    board_size = len(board)
    feature = np.zeros_like(board, dtype=np.float32)
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] == player:
                # iterates through each neighbour cell which would result in a bridge
                for direction_row, direction_col in BRIDGE_DIRECTIONS:
                    bridge_row = row + direction_row
                    bridge_col = col + direction_col
                    if not functionality.hexagon.in_bounds(board_size, cell=(bridge_row, bridge_col)):
                        continue
                    # branch if neighbour cell isn't occupied by player
                    if board[bridge_row][bridge_col] != player:
                        continue
                    # make sure none of the cells between are occupied
                    chain_connected = False
                    critical = False
                    critical_cells = []
                    cells_between = functionality.hexagon.cells_between(
                        cell=(row, col),
                        target=(bridge_row, bridge_col)
                    )
                    for cell_between_row, cells_between_col in cells_between:
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


def block(board: np.ndarray, player: int, opponent: int) -> np.ndarray:
    board_size = len(board)
    feature = np.zeros_like(board, dtype=np.float32)
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] == opponent:
                neighbour_cells = functionality.hexagon.in_bound_neighbours(board_size, cell=(row, col))
                for neighbour_row, neighbour_col in neighbour_cells:
                    if board[neighbour_row][neighbour_col] == EMPTY:
                        feature[neighbour_row][neighbour_col] += 1
    return feature


def winning_edges(board: np.ndarray, player: int, opponent: int) -> np.ndarray:
    feature = np.zeros_like(board, dtype=np.float32)
    if player == PLAYER_1:
        feature[0] = 1
        feature[-1] = 1
    else:
        feature[:, 0] = 1
        feature[:, -1] = 1
    return feature


def center_importance(board: np.ndarray, player: int, opponent: int) -> np.ndarray:
    board_size = len(board)
    distance_from_center = functionality.hexagon.get_distance_from_center(board_size)
    return distance_from_center
