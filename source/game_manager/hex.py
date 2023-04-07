# external libraries
import copy
from termcolor import colored 


class Hex:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.board = []
        self.player = 1
    
    def initialize_empty_board(self):
        self.board = self.initial_state()

    def initial_state(self) -> list[list[int]]:
        return [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]

    def play_move(self, move):
        self.board = apply_action_to_board(self.board, move, self.player)
        self.player = 2 if self.player == 1 else 1

    def terminal(self) -> bool:
        return terminal(self.board)
    
    def get_legal_actions(self) -> list[tuple[int, int]]:
        return get_legal_actions(self.board)
    
    def print_state(self, winning_path=None):
        print_state(self.board, winning_path)


def print_state(state: list[list[int]], winning_path=None):
    """
    Method to print the current game state
    """
    if winning_path is None:
        winning_path = []

    winning_path_set = set(winning_path)
    for i, row in enumerate(state):
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


def terminal(state: list[list[int]]) -> bool:
    def dfs(player, x, y, path):
        if (player == 1 and y == len(state) - 1) or (player == 2 and x == len(state) - 1):
            path.append((x, y))
            return path

        visited.add((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < len(state) and 0 <= ny < len(state)
                    and state[ny][nx] == player and (nx, ny) not in visited):
                new_path = dfs(player, nx, ny, path + [(x, y)])
                if new_path:
                    return new_path
        return False

    for player in [1, 2]:
        for i in range(len(state)):
            visited = set()
            start_x, start_y = (0, i) if player == 2 else (i, 0)
            if state[start_y][start_x] == player:
                winning_path = dfs(player, start_x, start_y, [])
                if winning_path:
                    return True

    for row in state:
        if 0 in row:
            return False
    return True
