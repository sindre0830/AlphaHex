from game_manager.Hex import *
from functionality import *
from mcts import *
from anet import *


class Main:
    def __init__(self):
        print("Starting AlphaHex...")
        self.board_size = 4
        self.game_manager = Hex(self.board_size)
        self.max_games = 10
        self.max_game_variations = 10
        self.exploration_constant = 3.0
        self.input_size = self.board_size * self.board_size
        self.hidden_size = 64  # The number of neurons in the hidden layer
        self.output_size = self.board_size * self.board_size  # The number of unique actions (assuming a maximum of 2 stones per pile)
        print("Ready to initialize MCTS...")
        self.anet = ANET(self.input_size, self.hidden_size, self.output_size)
        self.mcts = MCTS(self.game_manager, self.max_games, self.max_game_variations, None, self.exploration_constant, 1)

    def play_game(self):
        current_state = self.game_manager.initial_state
        player = 1

        while not self.game_manager.is_terminal(current_state):
            print(f"Player {player}'s turn (AI)")
            self.mcts.print_state(current_state)

            if player == 1:
                print("Thinking...")
                self.mcts.search(current_state)
                action = self.mcts.get_best_move()
                if action == None:
                    print("None action!")
            else:
                print("Enter your move (row and column separated by a space):")
                action = tuple(map(int, input().split()))

            if not self.game_manager.is_valid_action(current_state, action):
                print("Invalid move. Please try again.")
                continue

            current_state = self.game_manager.next_state(current_state, action, player)
            player = 2 if player == 1 else 1
            
            self.mcts.print_tree()

        self.mcts.print_state(current_state)
        winner = 2 if player == 1 else 1
        print(f"Player {winner} wins!")

if __name__ == "__main__":
    main = Main()
    main.play_game()
