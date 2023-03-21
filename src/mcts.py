from node import *
from game_manager.Nim import *
from functionality import *
from anet import *
import random
import queue
import numpy as np

class MCTS():
    def __init__(self, game_manager: Nim, max_games: int, max_game_variations: int, anet: ANET = None, exploration_constant: float = 1.0):
        self.game_manager = game_manager
        self.root = Node(state=game_manager.initial_state)
        self.max_games = max_games
        self.max_game_variations = max_game_variations
        self.anet = anet
        self.exploration_constant = exploration_constant
        
        
    def search(self) -> None:
        """
        Perform the MCTS search by running through the specified number of games, tree searching, simulating, and
        backpropagating
        """
        for i in range(self.max_games):
            print(f"Iteration: {i + 1}/{self.max_games}")
            leaf = self.tree_search(self.root)
            reward = self.simulate(leaf)
            self.backpropagate(leaf, reward)
            
        # Print the tree after each iteration
        self.print_tree()
    
    
    def tree_search(self, root: Node) -> Node:
        """
        Traverse the tree from the root node to a leaf node, selecting the best child node at each level
        """
        node = root
        while not self.game_manager.is_terminal(node.state):
            if len(node.children) == 0:
                self.expand(node)
            node = self.select_best_child(node)
        return node
    

    def evaluate_node(self, node: Node) -> float:
        """
        Evaluate the score of a node using the UCT algorithm
        """
        node.update_score(self.exploration_constant)
        return node.score
    
    
    def expand(self, node: Node) -> None:
        """
        Expand the tree by adding child nodes to a given node for each legal action in the current game state
        """
        legal_actions = self.game_manager.get_legal_actions(node.state)
        for action in legal_actions:
            next_state = self.game_manager.next_state(node.state, action)
            child_node = Node(state=next_state, parent=node)
            node.add_child(child_node)


    def select_best_child(self, node: Node) -> Node:
        """
        Select the child node with the highest score based on the UCT algorithm
        """        
        best_child = max(node.children, key=lambda child: self.evaluate_node(child))
        return best_child
    

    def simulate(self, node: Node) -> int:
        """        
        Simulate a game from the current game state, choosing random actions until the game is over, and return the winner
        """        
        state = node.state.copy()
        num_simulations = 0
        for _ in range(self.max_game_variations):
            num_simulations += 1
            while not self.game_manager.is_terminal(state):
                if (self.anet == None):
                    action = random.choice(self.game_manager.get_legal_actions(state))
                else:
                    legal_actions = self.game_manager.get_legal_actions(state)
                    action_values = self.anet.predict(state, legal_actions)
                    action = legal_actions[np.argmax(action_values)]                
                    state = self.game_manager.next_state(state, action)
        winner = node.player
        return winner
    

    def backpropagate(self, node: Node, reward: int) -> None:
        """
        Backpropagate the results of a game simulation up the tree, incrementing visit counts and win counts as necessary
        """
        current_node = node
        while current_node is not None:
            current_node.increment_visits()
            if current_node.player == reward:
                current_node.increment_wins()
            current_node.update_score(self.exploration_constant)
            current_node = current_node.parent


    def get_best_move(self) -> tuple[int, int]:
        """
        Get the best move to make from the root node by selecting the child node with the highest score and returning the
        action that led to that child node
        """
        best_child = self.select_best_child(self.root)
        for i, child in enumerate(self.root.children):
            if child == best_child:
                return self.game_manager.get_legal_actions(self.root.state)[i]
            
            
            
    def print_tree(self):
        """
        Method to print MCTS tree
        """
        print("MCTS Tree:")
        q = queue.Queue()
        q.put((self.root, 0))
        while not q.empty():
            node, depth = q.get()
            print(f"{'  ' * depth}Node: visits={node.visits}, wins={node.wins}, player={node.player}")
            for child in node.children:
                q.put((child, depth + 1))