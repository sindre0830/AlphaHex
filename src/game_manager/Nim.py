from functionality import *

class Nim():
    def __init__(self, rows = [1, 2, 3, 4]):
        self.winner: int = None
        self.rows: list[int] = rows
        self.min_take = 1
        self.max_take = max(self.rows)
    
    def move(self, move: int, player: int):
        row, take_amount = cantor_decode(move)
        if self.is_illegal(row, take_amount):
            return False
        self.update_state(row, take_amount)
        self.check_state(player)
        return True

    def update_state(self, row: int, take_amount: int):
        self.rows[row] -= take_amount
    
    def check_state(self, player):
        if sum(self.rows) <= 0:
            self.winner = 1 - player

    def is_illegal(self, row, take_amount):
        if row < 0 or row > len(self.rows):
            return True
        if take_amount < self.min_take or take_amount > self.max_take:
            return True
        if self.rows[row] < take_amount:
            return True
        return False

    def visualize_board(self):
        for row in self.rows:
            string = ""
            if row == 0:
                print()
            else:
                for _ in range(row):
                    string += "| "
                print(string)
