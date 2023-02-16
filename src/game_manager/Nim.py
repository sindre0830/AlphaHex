from functionality import *

class Nim():
    def __init__(self, rows: list[int] = [1, 2, 3, 4]):
        self.winner: int = None
        self.rows: list[int] = rows
        self.min_take: int = 1
        self.max_take: int = max(self.rows)
    
    def move(self, move: int, player: int):
        self.update_state(move)
        self.check_state(player)

    def update_state(self, move: int):
        row, take_amount = cantor_decode(move)
        self.rows[row] -= take_amount
    
    def check_state(self, player: int):
        if sum(self.rows) <= 0:
            self.winner = 1 - player
    
    def get_legal_moves(self):
        legal_moves: list[int] = []
        for row in range(len(self.rows)):
            for take_amount in range(self.rows[row]):
                legal_moves.append(cantor_encode(row, take_amount + 1))
        return legal_moves

    def visualize_board(self):
        for row in self.rows:
            string = ""
            if row == 0:
                print()
            else:
                for _ in range(row):
                    string += "| "
                print(string)
