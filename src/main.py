from game_manager.Nim import *
from functionality import *
from mcts import *
from anet import *
import numpy as np

class Main:
    def __init__(self):
        print("Starting AlphaHex...")
        self.initial_piles = [3, 4, 5]  
        self.game_manager = Nim(self.initial_piles)
        self.input_size = 3  # The number of piles in the Nim game
        self.hidden_size = 64  # The number of neurons in the hidden layer
        self.output_size = 6  # The number of unique actions (assuming a maximum of 2 stones per pile)
        self.max_games = 1000
        self.max_game_variations = 100
        self.exploration_constant = 1
        print("Ready to initialize ANET...")
        self.anet = ANET(input_size=2 * len(self.game_manager.initial_state), hidden_size=self.hidden_size, output_size=2 * len(self.game_manager.initial_state))
        print("Ready to initialize MCTS...")
        self.mcts = MCTS(self.game_manager, self.max_games, self.max_game_variations, self.anet, exploration_constant=self.exploration_constant)

    # Define MCTS parameters
    max_games = 10
    max_game_variations = 20
    exploration_constant = 1.0


    def play(self):
        current_state = self.game_manager.initial_state
        current_player = 0  # 0 for AI, 1 for human

        while not self.game_manager.is_terminal(current_state):
            print(f"Current state: {current_state}")
            self.print_board(current_state)
            if current_player == 0:
                # AI's turn
                print("AI's turn")
                self.mcts.search(current_state)
                best_move = self.mcts.get_best_move()
                print(f"AI's move: {best_move}")
            else:
                # Human's turn
                print("Your turn")
                legal_actions = self.game_manager.get_legal_actions(current_state)
                print(f"Legal actions: {legal_actions}")
                best_move = self.get_human_move(legal_actions)

            current_state = self.game_manager.next_state(current_state, best_move)
            current_player = 1 - current_player

        winner = "AI" if current_player == 1 else "You"
        print(f"Game over. {winner} won!")

    def get_human_move(self, legal_actions: list[tuple[int, int]]) -> tuple[int, int]:
        while True:
            pile_idx = int(input("Enter the pile index: "))
            count = int(input("Enter the number of stones to take: "))
            move = (pile_idx, count)
            if move in legal_actions:
                return move
            else:
                print("Invalid move. Please enter a valid move.")
                
    def print_board(self, state: list[int]) -> None:
        print("\nCurrent board:")
        for i, pile in enumerate(state):
            print(f"Pile {i}: {'*' * pile}")
        print()


if __name__ == "__main__":
    main = Main()
    main.play()