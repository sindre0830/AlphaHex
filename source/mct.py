# internal libraries
from node import Node
from game_manager.hex import (
    apply_action_to_board,
    get_legal_actions,
    terminal
)


class MCT:
    def __init__(self, exploration_constant: float = 1.0):
        self.root_node: Node = None
        self.game_board: list[list[int]] = None
        self.exploration_constant = exploration_constant
    
    def initialize_root_node(self, board: list[list[int]]):
        self.root_node = Node(board.copy())
    
    def update_game_board(self, board: list[list[int]]):
        self.game_board = board.copy()
    
    def tree_search(self) -> Node:
        node = self.root_node
        while not terminal(node.board):
            if (node.is_leaf_node()):
                return node
            node = max(node.children_nodes, key=lambda child_node: self.get_node_score(child_node))
            self.update_game_board(node.board)
        return node

    def node_expansion(self, node: Node):
        legal_actions = get_legal_actions(node.board)
        for action in legal_actions:
            next_board = apply_action_to_board(node.board, action, node.player)
            next_player = 2 if node.player == 1 else 1
            child_node = Node(next_board, player=next_player, parent_node=node)
            node.add_child(child_node, action)
        self.game_board = node.board.copy()
    
    def get_node_score(self, node: Node):
        node.update_score(self.exploration_constant)
        return node.score
