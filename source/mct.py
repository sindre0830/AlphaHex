# internal libraries
from node import Node
from anet import ANET
from game_manager.hex import (
    apply_action_to_board,
    get_legal_actions,
    terminal,
    get_winner
)
# external libraries
import random
import copy


class MCT:
    def __init__(self, exploration_constant: float = 1.0):
        self.root_node: Node = None
        self.game_board: list[list[int]] = None
        self.exploration_constant = exploration_constant
        self.cached_actions = {}
        self.turn = 0
    
    def set_root_node(self, board: list[list[int]], player: int, reset_turn = False):
        self.root_node = Node(copy.deepcopy(board), player)
        self.cached_actions.clear()
        if reset_turn:
            self.turn = 0
        else:
            self.turn += 1
    
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
        turn = self.turn
        while not terminal(board):
            legal_actions = get_legal_actions(board)
            probability_distribution = anet.predict(legal_actions, state=(board, player, turn))
            action = random.choices(population=legal_actions, weights=probability_distribution, k=1)[0]
            board = apply_action_to_board(board, action, player)
            player = 2 if player == 1 else 1
            turn += 1
        return get_winner(board)
    
    def backpropagate(self, node: Node, score: int):
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            if current_node.player == score:
                current_node.wins += 1
            current_node.update_score(self.exploration_constant)
            current_node = current_node.parent_node
    
    def get_probability_distribution(self, anet: ANET, legal_actions: list[tuple[int, int]], state: tuple[list[list[int]], int, int]) -> list[float]:
        key = (tuple(legal_actions), tuple(map(tuple, state[0])), state[1])
        if key in self.cached_actions:
            return self.cached_actions[key]
        else:
            probability_distribution = anet.predict(legal_actions, state)
            self.cached_actions[key] = probability_distribution
            return probability_distribution
