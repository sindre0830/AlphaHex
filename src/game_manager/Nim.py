from functionality import *

class Nim():
    def __init__(self, rows = [1, 2, 3, 4]):
        self.winner: int = None
        self.rows: list[int] = rows
        self.min_take = 1
        self.max_take = max(self.rows)
    
    def move(self, move: int, player: int):
        row, take_amount = cantor_decode(move)
        self.update_state(row, take_amount)
        self.check_state(player)
        return True

    def update_state(self, row: int, take_amount: int):
        self.rows[row] -= take_amount
    
    def check_state(self, player):
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
