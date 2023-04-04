# internal libraries
from node import Node


class MCT:
    def __init__(self):
        self.root_node: Node = None
        self.game_board: list[list[int]] = None
    
    def initialize_root_node(self, board: list[list[int]]):
        self.root_node = Node(board.copy())

    def set_game_board_from_root(self):
        self.game_board = self.root_node.board.copy()
