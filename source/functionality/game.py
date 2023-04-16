# external libraries
import math


def action_to_index(action: tuple[int, int], width: int) -> int:
    row, column = action
    return (row * width) + column


def index_to_action(index: int, width: int) -> tuple[int, int]:
    row = math.floor(index / width)
    column = index - (row * width)
    return (row, column)


def opposite_player(current_player: int) -> int:
    return 2 if current_player == 1 else 1
