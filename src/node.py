import math

class Node():
    def __init__(self, state: list[int], parent = None, player=1, child_actions = [], legal_actions=[]):
        self.Q: float = 0.0 # exploitation term
        self.U: float = 0.0 # exploration term
        self.score: float = 0.0
        self.state: list[list[int]] = state
        self.parent: Node = parent
        self.children: list[Node]= []
        self.wins: int = 0
        self.visits: int = 0
        self.player = player
        self.child_actions: list[tuple[int, int]] = child_actions
        self.legal_actions = legal_actions
        
        
    def increment_wins(self) -> None:
        self.wins += 1


    def increment_visits(self) -> None:
        self.visits += 1


    def add_child(self, node, action) -> None:
        self.children.append(node)
        self.child_actions.append(action)
        
        
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


    def get_distribution(self, max_actions=20, board_size=5):
        """
        Get the distribution of visit counts for all child nodes of the current node.
        :param max_actions: The maximum number of actions.
        :param board_size: The size of the board.
        :return: The distribution of visit counts for all child nodes.
        """
        distribution = [0] * max_actions
        total_visits = sum(child.visits for child in self.children)
        
        for child, action in zip(self.children, self.child_actions):
            if action is not None:
                action_index = action[1] * board_size + action[0]
                if action_index < max_actions:
                    distribution[action_index] = child.visits / total_visits
        
        return distribution