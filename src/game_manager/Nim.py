from functionality import *

class Nim():
    def __init__(
        self,
        initial_state: list[list[int]] = [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ]
    ):
        self.initial_state: list[list[int]] = initial_state
        self.min_take: int = 1
        self.max_take: int = len(initial_state)

    def update_state(self, state: list[list[int]], move: int):
        row_index, take_amount = cantor_decode(move)
        row: list[int] = state[row_index].copy()
        row.reverse()

        for _ in range(take_amount):
            for i in range(len(row)):
                if row[i] == 1:
                    row[i] = 0
                    break
        row.reverse()
        state[row_index] = row
        return state
    
    def get_win_state(self, state: list[list[int]], player: int):
        total_pieces = 0
        for row in state:
            total_pieces += sum(row)
        if total_pieces <= 0:
            return 1 - player
        else:
            return None
    
    def get_legal_moves(self, state: list[list[int]]):
        legal_moves: list[int] = []
        for row in range(len(state)):
            for take_amount in range(sum(state[row])):
                legal_moves.append(cantor_encode(row, take_amount + 1))
        return legal_moves

    def visualize_board(self, state: list[list[int]]):
        board_str = ""
        for row in state:
            row_str = ""
            for elem in row:
                if elem == 0: 
                    row_str += "0 "
                else:
                    row_str += "1 "
            board_str += row_str + "\n"
        print(board_str)
