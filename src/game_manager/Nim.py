class Nim():
    def __init__(self, rows = [1, 2, 3, 4]):
        self.finished: bool = False
        self.rows: list[int] = rows
        self.min_take = 1
        self.max_take = max(self.rows)
