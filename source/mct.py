# internal libraries
from node import Node
from anet import ANET
from game_manager.hex import (
    apply_action_to_board,
    get_legal_actions,
    terminal
)
# external libraries
import random
import copy


class MCT:
    def __init__(self, exploration_constant: float = 1.0):
        self.root_node: Node = None
        self.game_board: list[list[int]] = None
        self.exploration_constant = exploration_constant
    
    def set_root_node(self, board: list[list[int]], player: int):
        self.root_node = Node(copy.deepcopy(board), player)
    
    def update_game_board(self, board: list[list[int]]):
        self.game_board = copy.deepcopy(board)
    
    def tree_search(self) -> Node:
        node = self.root_node
        while not terminal(node.board):
            if (node.is_leaf_node()):
                return node
            node = max(node.children_nodes, key=lambda child_node: child_node.get_score(self.exploration_constant))
            self.update_game_board(node.board)
        return node

    def node_expansion(self, node: Node):
        legal_actions = get_legal_actions(node.board)
        for action in legal_actions:
            next_board = apply_action_to_board(node.board, action, node.player)
            next_player = 2 if node.player == 1 else 1
            child_node = Node(next_board, player=next_player, parent_node=node)
            node.add_child(child_node, action)
    
    def leaf_evaluation(self, anet: ANET, node: Node):
        board = copy.deepcopy(node.board)
        player = node.player
        winner = 2 if player == 1 else 1
        while not terminal(board):
            legal_actions = get_legal_actions(board)
            action_values = anet.predict(legal_actions, state=(board, player))
            action = random.choices(population=legal_actions, weights=action_values, k=1)[0]
            board = apply_action_to_board(board, action, player)
            winner = player
            player = 2 if player == 1 else 1
        return winner
    
    def backpropagate(self, node: Node, score: int):
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            if current_node.player == score:
                current_node.wins += 1
            current_node.update_score(self.exploration_constant)
            current_node = current_node.parent_node
