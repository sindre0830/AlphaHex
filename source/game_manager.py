# internal libraries
from functionality.board import (
    legal_actions
)
# external libraries
from termcolor import colored
import numpy as np


class GameManager:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.board = self.empty_board()
        self.player = 1
    
    def set_state(self, board: np.ndarray, player = 1):
        self.board = np.copy(board)
        self.player = player

    def empty_board(self) -> np.ndarray:
        return np.zeros(shape=(self.board_size, self.board_size), dtype=np.int8)

    def play_move(self, move):
        apply_action_to_board(self.board, move, self.player)
        self.player = 2 if self.player == 1 else 1

    def terminal(self) -> bool:
        return terminal(self.board)
    
    def legal_actions(self) -> list[tuple[int, int]]:
        return legal_actions(self.board)
    
    def print_state(self, winning_path=None):
        print_state(self.board, winning_path)
    
    def get_winner(self) -> int:
        return get_winner(self.board)


def print_state(board: np.ndarray, winning_path=None):
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


def apply_action_to_board(board: np.ndarray, action: tuple[int, int], player: int):
    row, col = action
    board[row][col] = player


def terminal(board: np.ndarray) -> bool:
    return get_winner(board) != -1


def get_winner(board: np.ndarray) -> int:
    board_size = len(board)
    def dfs(player: int, x: int, y: int, path: list):
        if (player == 1 and y == board_size - 1) or (player == 2 and x == board_size - 1):
            path.append((x, y))
            return path
        visited.add((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < board_size and 0 <= ny < board_size and board[ny][nx] == player and (nx, ny) not in visited):
                new_path = dfs(player, nx, ny, path + [(x, y)])
                if new_path:
                    return new_path
        return False
    for player in [1, 2]:
        for i in range(board_size):
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
