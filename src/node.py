import math

class Node():
    def __init__(self, state: list[int], parent = None):
        self.Q: float = 0.0 # exploitation term
        self.U: float = 0.0 # exploration term
        self.score: float = 0.0
        self.state: list[list[int]] = state
        self.parent: Node = parent
        self.children: list[Node]= []
        self.wins: int = 0
        self.visits: int = 0
        self.player: int = 0 if self.parent is None else 1 - self.parent.player
        
    def increment_wins(self) -> None:
        self.wins += 1

    def increment_visits(self) -> None:
        self.visits += 1

    def add_child(self, node) -> None:
        self.children.append(node)
        
    def update_score(self, exploration_constant: float) -> None:
        self.__update_Q()
        self.__update_U(exploration_constant)
        self.score = self.Q + self.U
        
    def __update_Q(self) -> None:
        """
        This method updates the win ratio for a node.
        """
        if self.visits == 0:
            self.Q = 0.0
        else:
            self.Q = self.wins / self.visits
        
    def __update_U(self, exploration_constant: float = 1.0) -> None:
        """
        This method updates u our exploration value, through default tree policy, UCT.
        https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation
        """
        if self.parent is None:
            self.U = float('inf')
        elif self.visits == 0 or self.parent.visits == 0:
            self.U = float('inf')
        else:
            self.U = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

