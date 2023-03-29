from node import *
from game_manager.hex import *
from functionality import *
from anet import *
import random
import queue
import matplotlib.pyplot as plt
import numpy as np

class MCTS():
    def __init__(self, game_manager: Hex, max_games: int, max_game_variations: int, anet: ANET = None, exploration_constant: float = 1.0, starting_player=1, epsilon=0.0):
        self.game_manager = game_manager  # The game manager, which handles game logic
        self.root = Node(state=game_manager.initial_state)  # The root of the MCTS tree
        self.max_games = max_games  # The maximum number of games to search through
        self.max_game_variations = max_game_variations  # The maximum number of game variations to simulate
        self.anet = anet  # The Action-Value Neural Network (ANET)
        self.exploration_constant = exploration_constant  # The exploration constant for the UCT algorithm
        self.starting_player = starting_player
        self.epsilon = epsilon
        


    def search(self, current_state: list[list[int]]) -> None:
        # Perform the MCTS search by running through the specified number of games, tree searching, simulating, and backpropagating
        self.root = Node(state=current_state)  # Update the root with the current state
        for i in range(self.max_games):
            print(f"\nIteration: {i + 1}/{self.max_games}")
            leaf = self.tree_search(self.root)
            reward = self.simulate(leaf)
            self.backpropagate(leaf, reward)


    def tree_search(self, root: Node) -> Node:
        # Traverse the tree from the root node to a leaf node, selecting the best child node at each level
        # node = root
        # if not self.game_manager.is_terminal(node.state):
        #     if len(node.children) == 0:
        #         self.expand(node)
        #     leaf = self.select_best_child(node)
        # return leaf
    
        node = root
        while not self.game_manager.is_terminal(node.state):
            if len(node.children) == 0:
                self.expand(node)
                return node
            node = self.select_best_child(node)
        return node


    def evaluate_node(self, node: Node) -> float:
        # Evaluate the score of a node using the UCT algorithm
        node.update_score(self.exploration_constant)
        return node.score


    def expand(self, node: Node) -> None:
        # Expand the tree by adding child nodes to a given node for each legal action in the current game state
        legal_actions = self.game_manager.get_legal_actions(node.state)
        for action in legal_actions:
            next_state = self.game_manager.next_state(node.state, action, node.player)
            next_player = 2 if node.player == 1 else 1
            next_legal_actions = self.game_manager.get_legal_actions(next_state)
            child_node = Node(state=next_state, parent=node, player=next_player, child_actions=[action], legal_actions=next_legal_actions)
            node.add_child(child_node, action)
        print("Finished expanding, leaf found.")


    def select_best_child(self, node: Node) -> Node:
        # Select the child node with the highest score based on the UCT algorithm
        best_child = max(node.children, key=lambda child: self.evaluate_node(child))
        return best_child
    
    
    def get_best_valid_action(self, state, legal_actions, player):
        action_values = self.anet.predict(state, legal_actions)
        sorted_actions = sorted(zip(action_values, legal_actions), key=lambda x: x[0], reverse=True)

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        for action_value, action in sorted_actions:
            if action in legal_actions:
                return action
        return None  # Return None if no valid action is found


    def simulate(self, node: Node) -> int:
        print("Starting simulation from this state:")
        # Simulate a game from the current game state, choosing random actions until the game is over, and return the winner
        num_simulations = 0
        initial_player = node.player
        initial_state = [row.copy() for row in node.state] 
        for i in range(self.max_game_variations):
            print(f"Game variation: {i + 1}")
            state = [row.copy() for row in initial_state]  # Move this line inside the loop
            player = initial_player
            num_simulations += 1
            while not self.game_manager.is_terminal(state):
                legal_actions = self.game_manager.get_legal_actions(state)
                if not legal_actions:
                    print("No legal moves")
                    break
                if (self.anet == None):
                    action = random.choice(legal_actions)
                else:
                    action = self.get_best_valid_action(state, legal_actions, player)
                    if action is None:
                        print("No valid action found in simulate()")
                        break
                state = self.game_manager.next_state(state, action, player)
                player = 2 if player == 1 else 1
            player_that_won=2 if player==1 else 1
            if self.game_manager.is_terminal(state):
                winner = player_that_won
                break
        # winner = node.player
        print("Finished simulating")
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
        action that led to that child node, considering only legal moves. If the best move is not valid, choose the next
        best move.
        """
        legal_actions = self.game_manager.get_legal_actions(self.root.state)
        print(f"Legal actions for current state: {legal_actions}")
    
        child_action_pairs = [(child, action) for child, action in zip(self.root.children, self.root.child_actions)]
        sorted_child_action_pairs = sorted(child_action_pairs, key=lambda x: x[0].score, reverse=True)


        # print(f"get_best_move() - Choosing from these {len(sorted_child_action_pairs)} moves:")
        # for child, action in sorted_child_action_pairs:
        #     print(child.score, action, child.wins, child.visits)
            
        # print(f"Board looks like this for {sorted_child_action_pairs[0][1]}: ")
        # self.game_manager.print_state(sorted_child_action_pairs[0][0].parent.state)
            
        for child, action in sorted_child_action_pairs:
            if self.game_manager.is_valid_action(child.parent.state, action):
                # print("Choosing: " + str(action))
                return action
        # If no legal moves are found (should not happen in a normal game), return None
        print("No move found.")
        return None