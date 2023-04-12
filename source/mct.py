# internal libraries
from functionality import (
    opposite_player
)
from node import Node
from anet import ANET
from game_manager.hex import (
    Hex,
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
        self.turn = 0
        self.game_board_history: list[list[list[int]]] = []
    
    def set_root_node(self, board: list[list[int]], player: int, reset_turn = False):
        self.root_node = Node(copy.deepcopy(board), player)
        if reset_turn:
            self.turn = 0
            self.game_board_history = []
        else:
            self.turn += 1
        self.game_board_history.append(copy.deepcopy(board))
    
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
            next_player = opposite_player(node.player)
            child_node = Node(next_board, player=next_player, parent_node=node)
            node.add_child(child_node, action)
    
    def leaf_evaluation(self, anet: ANET, node: Node):
        local_game_manager = Hex(board_size=len(node.board))
        local_game_manager.set_state(node.board, node.player)
        turn = self.turn
        while not local_game_manager.terminal():
            legal_actions = local_game_manager.get_legal_actions()
            state = (local_game_manager.board, local_game_manager.player, turn)
            probability_distribution = anet.predict(legal_actions, state)
            action = random.choices(population=legal_actions, weights=probability_distribution, k=1)[0]
            local_game_manager.play_move(action)
            turn += 1
        return local_game_manager.get_winner()
    
    def backpropagate(self, node: Node, score: int):
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            if current_node.player == score:
                current_node.wins += 1
            current_node.update_score(self.exploration_constant)
            current_node = current_node.parent_node
