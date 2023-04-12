# internal libraries
from functionality import (
    opposite_player
)
# external libraries
import copy
from termcolor import colored


class Hex:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.board = []
        self.player = 1
    
    def set_state(self, board: list[list[int]], player = 1):
        self.board = copy.deepcopy(board)
        self.player = player

    def empty_board(self) -> list[list[int]]:
        return [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]

    def play_move(self, move):
        self.board = apply_action_to_board(self.board, move, self.player)
        self.player = opposite_player(self.player)

    def terminal(self) -> bool:
        return terminal(self.board)
    
    def get_legal_actions(self) -> list[tuple[int, int]]:
        return get_legal_actions(self.board)
    
    def print_state(self, winning_path=None):
        print_state(self.board, winning_path)
    
    def get_winner(self) -> int:
        return get_winner(self.board)


def print_state(board: list[list[int]], winning_path=None):
    """
    Method to print the current game board
    """
    if winning_path is None:
        winning_path = []

    winning_path_set = set(winning_path)
    for i, row in enumerate(board):
        print(' ' * i, end='')
        colored_row = []
        for j, cell in enumerate(row):
            if (j, i) in winning_path_set:
                colored_row.append(colored(str(cell), "red"))
            else:
                colored_row.append(str(cell))
        print(' '.join(colored_row))


def apply_action_to_board(board: list[list[int]], action: tuple[int, int], player: int) -> list[list[int]]:
    x, y = action
    next_board = copy.deepcopy(board)
    next_board[y][x] = player
    return next_board


def get_legal_actions(board: list[list[int]]) -> list[tuple[int, int]]:
    return [(x, y) for x in range(len(board)) for y in range(len(board)) if board[y][x] == 0]


def terminal(board: list[list[int]]) -> bool:
    if (get_winner(board) != -1):
        return True
    else:
        return False


def get_winner(board: list[list[int]]) -> int:
    board_width = len(board)
    def dfs(player: int, x: int, y: int, path: list):
        if (player == 1 and y == board_width - 1) or (player == 2 and x == board_width - 1):
            path.append((x, y))
            return path
        visited.add((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < board_width and 0 <= ny < board_width and board[ny][nx] == player and (nx, ny) not in visited):
                new_path = dfs(player, nx, ny, path + [(x, y)])
                if new_path:
                    return new_path
        return False
    for player in [1, 2]:
        for i in range(board_width):
            visited = set()
            start_x, start_y = (0, i) if player == 2 else (i, 0)
            if board[start_y][start_x] == player:
                winning_path = dfs(player, start_x, start_y, [])
                if winning_path:
                    return player
    for row in board:
        if 0 in row:
            return -1
    return 0
