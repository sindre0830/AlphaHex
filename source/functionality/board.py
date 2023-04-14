# internal libraries
from constants import (
    NEIGHBOUR_DIRECTIONS,
    EMPTY
)
# external libraries
import numpy as np


def in_bound_neighbours(board_size: int, cell: tuple[int, int]) -> list[tuple[int, int]]:
    cell_row, cell_col = cell
    neighbours: list[tuple[int, int]] = []
    for direction_row, direction_col in NEIGHBOUR_DIRECTIONS:
        neighbour = (direction_row + cell_row, direction_col + cell_col)
        if in_bounds(board_size, neighbour):
            neighbours.append(neighbour)
    return neighbours


def in_bounds(board_size: int, cell: tuple[int, int]) -> bool:
    row, col = cell
    row_in_bounds = row >= 0 and row < board_size
    col_in_bounds = col >= 0 and col < board_size
    return row_in_bounds and col_in_bounds


def cells_between(cell: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
    cell_row, cell_col = cell
    neighbours: list[tuple[int, int]] = []
    for direction_row, direction_col in NEIGHBOUR_DIRECTIONS:
        neighbour = (direction_row + cell_row, direction_col + cell_col)
        if adjacent(neighbour, target):
            neighbours.append(neighbour)
    return neighbours


def adjacent(cell: tuple[int, int], target: tuple[int, int]) -> bool:
    cell_row, cell_col = cell
    target_row, target_col = target
    for direction_row, direction_col in NEIGHBOUR_DIRECTIONS:
        neighbour_row = direction_row + cell_row
        neighbour_col = direction_col + cell_col
        if neighbour_row == target_row and neighbour_col == target_col:
            return True
    return False


def get_distance_from_center(board_size: int) -> np.ndarray:
    distance_from_center = np.zeros(shape=(board_size, board_size), dtype=np.float32)
    center_row = board_size // 2
    center_col = board_size // 2
    center_cell = (center_row, center_col)
    border_cells: list[tuple[int, int]] = [center_cell]
    buffer_border_cells: list[tuple[int, int]] = []
    value = 1
    distance_from_center[center_row][center_col] = value
    while len(border_cells) != 0:
        value += 1
        for cell in border_cells:
            for neighbour_cell in in_bound_neighbours(board_size, cell):
                neighbour_row, neighbour_col = neighbour_cell
                if distance_from_center[neighbour_row][neighbour_col] == EMPTY:
                    distance_from_center[neighbour_row][neighbour_col] = value
                    buffer_border_cells.append(neighbour_cell)
        border_cells = buffer_border_cells.copy()
        buffer_border_cells = []
    return distance_from_center


def legal_actions(board: list[list[int]]) -> list[tuple[int, int]]:
    board_size = len(board)
    actions: list[tuple[int, int]] = []
    for row in range(board_size):
        for col in range(board_size):
            if (board[row][col] == EMPTY):
                actions.append((row, col))
    return actions


def illegal_actions(board: list[list[int]]) -> list[tuple[int, int]]:
    board_size = len(board)
    actions: list[tuple[int, int]] = []
    for row in range(board_size):
        for col in range(board_size):
            if (board[row][col] != EMPTY):
                actions.append((row, col))
    return actions
