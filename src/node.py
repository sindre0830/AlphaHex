class Node():
    def __init__(self, state: list[int], parent = None):
        self.parent: Node = parent
        self.state: list[list[int]] = state
        self.children: list[Node]= []
        self.wins: int = 0
        self.visits: int = 0
        self.player: int = 0 if self.parent is None else 1 - self.parent.player # Be aware of this if using several models
        
    def increment_wins(self):
        self.wins += 1

    def increment_visits(self):
        self.visits += 1

    def add_child(self, node):
        self.children.append(node)
