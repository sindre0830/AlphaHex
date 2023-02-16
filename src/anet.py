import random

class ANET():
    def __init__(self):
        pass

    def get_move(self, player: int, legal_moves: list[int]):
        return random.choice(legal_moves)
