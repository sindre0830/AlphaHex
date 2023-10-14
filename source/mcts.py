# internal libraries
from node import Node
from anet import ANET
from state_manager import StateManager
import functionality.data
# external libraries
import numpy as np


class MCTS:
    def __init__(self, exploration_constant: float, greedy_epsilon: float = None):
        self.root_node: Node = None
        self.max_depth = 1
        self.exploration_constant = exploration_constant
        self.greedy_epsilon = greedy_epsilon

    def dynamic_depth(self, round: int):
        if round >= 20:
            self.max_depth = 5
        elif round >= 15:
            self.max_depth = 4
        elif round >= 10:
            self.max_depth = 3
        else:
            self.max_depth = 2

    def dynamic_greedy_epsilon(self, iteration: int, max_iterations: int, max_epsilon: float, min_epsilon: float):
        if self.greedy_epsilon is not None:
            self.greedy_epsilon = max_epsilon - (max_epsilon - min_epsilon) * (iteration / max_iterations)

    def set_root_node(self, state: StateManager):
        self.root_node = Node(state)

    def tree_search(self) -> Node:
        node = self.root_node
        while not node.state.terminal():
            if (node.is_leaf_node() or node.depth() >= self.max_depth):
                return node
            node = max(node.children_nodes, key=lambda child_node: child_node.get_score(self.exploration_constant))
        return node

    def node_expansion(self, node: Node):
        if node.depth() >= self.max_depth:
            return
        for action in node.state.legal_actions():
            node.add_child(Node(node.state, action, parent_node=node))

    def leaf_evaluation(self, anet: ANET, leaf: Node) -> tuple[Node, int]:
        if leaf.depth() >= self.max_depth or leaf.state.terminal():
            node = leaf
        else:
            state = (leaf.state.grid, leaf.state.player)
            probability_distribution = anet.predict(state, filter_actions=leaf.state.illegal_actions())
            best_action = functionality.data.index_to_action(np.argmax(probability_distribution), leaf.state.grid_size)
            for child in leaf.children_nodes:
                if child.state.action == best_action:
                    node = child
                    break
        # perform rollout on child node
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
        return node, local_state.determine_winner(), local_state.round()

    def backpropagate(self, node: Node, winner: int, round: int):
        current_node = node
        # min_rounds = 1 + node.state.grid_size * 2
        # max_rounds = 1 + node.state.total_possible_moves()
        while current_node is not None:
            current_node.visits += 1
            if current_node.state.player == winner:
                # current_node.wins += (round - min_rounds) / (max_rounds - min_rounds) * (0.5 - 1.5) + 1.5
                current_node.wins -= 1
            current_node = current_node.parent_node
