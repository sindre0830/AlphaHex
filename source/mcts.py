# internal libraries
from node import Node
from anet import ANET
from state_manager import StateManager


class MCTS:
    def __init__(self, exploration_constant: float, greedy_epsilon: float = None):
        self.root_node: Node = None
        self.exploration_constant = exploration_constant
        self.greedy_epsilon = greedy_epsilon
    
    def dynamic_greedy_epsilon(self, iteration: int, max_iterations: int, max_epsilon: float, min_epsilon: float):
        if self.greedy_epsilon is not None:
            self.greedy_epsilon = max_epsilon - (max_epsilon - min_epsilon) * (iteration / max_iterations)
    
    def set_root_node(self, state: StateManager):
        self.root_node = Node(state)
    
    def tree_search(self) -> Node:
        node = self.root_node
        while not node.state.terminal():
            if (node.is_leaf_node()):
                return node
            node = max(node.children_nodes, key=lambda child_node: child_node.get_score(self.exploration_constant))
        return node

    def node_expansion(self, node: Node):
        for action in node.state.legal_actions():
            node.add_child(Node(node.state, action, parent_node=node))
    
    def leaf_evaluation(self, anet: ANET, node: Node):
        local_state = StateManager()
        local_state.copy_state(node.state)
        while not local_state.terminal():
            state = (local_state.grid, local_state.player)
            probability_distribution = anet.predict(state, filter_actions=local_state.illegal_actions())
            local_state.apply_action_from_distribution(
                probability_distribution,
                deterministic=False, 
                greedy_epsilon=self.greedy_epsilon
            )
        return local_state.determine_winner()
    
    def backpropagate(self, node: Node, winner: int):
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            if current_node.state.player == winner:
                current_node.wins += 1
            current_node = current_node.parent_node
