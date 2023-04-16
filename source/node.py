# internal libraries
from functionality.game import (
    action_to_index
)
from state_manager import StateManager
# external libraries
import math
import numpy as np


class Node():
    def __init__(self, state: StateManager, action: tuple[int, int] = None, parent_node=None):
        self.state = StateManager()
        self.state.copy_state(state)
        if action is not None:
            self.state.apply_action(action)
        self.parent_node: Node = parent_node
        self.children_nodes: list[Node] = []
        self.score = 0.0
        self.wins = 0
        self.visits = 0
    
    def is_leaf_node(self) -> bool:
        return len(self.children_nodes) == 0

    def add_child(self, node):
        self.children_nodes.append(node)
    
    def get_score(self, exploration_constant: float) -> float:
        self.update_score(exploration_constant)
        return self.score

    def update_score(self, exploration_constant: float):
        self.score = self.get_q() + self.get_u(exploration_constant)

    def get_q(self) -> float:
        if (self.visits == 0):
            return 0
        return self.wins / self.visits
    
    def get_u(self, exploration_constant: float) -> float:
        if self.parent_node is None or self.visits == 0 or self.parent_node.visits == 0:
            return float('inf')
        return exploration_constant * math.sqrt(math.log(self.parent_node.visits) / self.visits)

    def visit_distribution(self) -> np.ndarray:
        distribution = np.zeros(shape=self.state.total_possible_moves(), dtype=np.float32)
        total_visits = sum(child_node.visits for child_node in self.children_nodes)
        for child_node in self.children_nodes:
            if child_node.state.action is not None:
                action_index = action_to_index(child_node.state.action, width=self.state.grid_size)
                distribution[action_index] = child_node.visits / total_visits
        return distribution
