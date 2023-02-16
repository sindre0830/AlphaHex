from node import *
from game_manager.nim import *
from functionality import *
import random
import math

class MCTS():
    def __init__(self, game_manager: Nim, max_games: int, max_game_variations: int, c: float = math.sqrt(2)):
        self.game_manager = game_manager
        self.root = Node(state=game_manager.initial_state)
        self.max_games = max_games
        self.max_game_variations = max_game_variations
        self.c = c # Exploration parameter
    
    def run(self):
        pass

    #  Traversing the tree from the root to a leaf node by using the tree policy.
    def tree_search(self):
        pass
    
    def evaluate_node(self, node: Node):
        return node.wins/node.visits + self.c * math.sqrt((math.log(node.parent.wins))/node.visits)

    # Generating k or all child states of a parent state, and then connecting the tree node housing
    # the parent state (a.k.a. parent node) to the nodes housing the child states (a.k.a. child nodes).
    def node_expansion(self, parent: Node):
        legal_moves = self.game_manager.get_legal_moves(parent.state)
        random.shuffle(legal_moves) # use ANET
        for move in legal_moves[:self.max_game_variations]:
            new_state = self.game_manager.update_state(parent.state.copy(), move)
            leaf = Node(new_state, parent)
            parent.add_child(leaf)

    # Estimating the value of a leaf node in the tree by doing a rollout simulation using the default
    # policy from the leaf node's state to a final state.
    def leaf_evaluation(self):
        pass

    # Passing the evaluation of a final state back up the tree, updating relevant data (see course
    # lecture notes) at all nodes and edges on the path from the final state to the tree root.
    def backpropagation(self):
        pass