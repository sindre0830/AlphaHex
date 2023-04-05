# internal libraries
from functionality import (
    action_to_index
)
# external libraries
import math


class Node():
    def __init__(self, board: list[list[int]], player=1, parent_node=None):
        self.board = board
        self.player = player
        self.parent_node: Node = parent_node
        self.children_nodes: list[Node] = []
        self.children_nodes_actions: list[tuple[int, int]] = []
        self.score = 0.0
        self.wins = 0
        self.visits = 0
    
    def is_leaf_node(self) -> bool:
        return len(self.children_nodes) == 0

    def add_child(self, node, action) -> None:
        self.children_nodes.append(node)
        self.children_nodes_actions.append(action)
    
    def get_score(self, exploration_constant: float) -> float:
        self.update_score(exploration_constant)
        return self.score

    def update_score(self, exploration_constant: float):
        self.score = self.get_q() + self.get_u(exploration_constant)

    def get_q(self) -> float:
        if (self.visits == 0):
            return 0.0
        else:
            return self.wins / self.visits
    
    def get_u(self, exploration_constant: float) -> float:
        if self.parent_node is None:
            return float('inf')
        elif self.visits == 0 or self.parent_node.visits == 0:
            return float('inf')
        else:
            return exploration_constant * math.sqrt(math.log(self.parent_node.visits) / self.visits)

    def visit_distribution(self):
        distribution = [0.0] * (len(self.board) * len(self.board))
        total_visits = sum(child_node.visits for child_node in self.children_nodes)
        for child_node, action in zip(self.children_nodes, self.children_nodes_actions):
            if action is not None:
                action_index = action_to_index(action, width=len(self.board))
                distribution[action_index] = child_node.visits / total_visits
        return distribution
