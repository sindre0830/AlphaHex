from typing import List, Tuple

class Hex:
    def __init__(self, board_size: int):
        self.board_size = board_size

    @property
    def initial_state(self) -> List[List[int]]:
        return [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]

    def is_terminal(self, state: List[List[int]]) -> bool:
        def dfs(player, x, y, path):
            if (player == 1 and y == len(state) - 1) or (player == 2 and x == len(state) - 1):
                path.append((x, y))
                print(f"Winning path for player {player}: {path}")
                return True

            visited.add((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < len(state) and 0 <= ny < len(state)
                        and state[ny][nx] == player and (nx, ny) not in visited):
                    if dfs(player, nx, ny, path + [(x, y)]):
                        return True
            return False

        for player in [1, 2]:
            for i in range(len(state)):
                visited = set()
                start_x, start_y = (0, i) if player == 2 else (i, 0)
                if state[start_y][start_x] == player and dfs(player, start_x, start_y, []):
                    return True

        for row in state:
            if 0 in row:
                return False
        return True


    def get_legal_actions(self, state: List[List[int]]) -> List[Tuple[int, int]]:
        legal_actions = [(x, y) for x in range(self.board_size) for y in range(self.board_size) if state[y][x] == 0]
        return legal_actions

    def next_state(self, state: List[List[int]], action: Tuple[int, int], player: str) -> List[List[int]]:
        x, y = action
        next_state = [row.copy() for row in state]
        next_state[y][x] = player
        return next_state

    def is_valid_action(self, state: List[List[int]], action: Tuple[int, int]) -> bool:
        x, y = action
        return 0 <= x < self.board_size and 0 <= y < self.board_size and state[y][x] == 0
