from game_manager.Nim import *
from functionality import *
from mcts import *
from anet import *
import numpy as np

def main():
    print("Starting AlphaHex...")
    # Define ANET parameters
    input_size = 3  # The number of piles in the Nim game
    hidden_size = 64  # The number of neurons in the hidden layer
    output_size = 6  # The number of unique actions (assuming a maximum of 2 stones per pile)

    print("Ready to initialize ANET...")
    # Initialize the ANET
    anet = ANET(input_size, hidden_size, output_size)

    # Define the initial state for the Nim game
    initial_piles = [3, 4, 5]
    nim_game_manager = Nim(initial_piles)

    # Define MCTS parameters
    max_games = 10
    max_game_variations = 20
    exploration_constant = 1.0

    print("Ready to initialize MCTS...")
    # Initialize the MCTS
    mcts = MCTS(game_manager=nim_game_manager, max_games=max_games, max_game_variations=max_game_variations, anet=anet, exploration_constant=exploration_constant)

    # Run the MCTS algorithm
    mcts.search()

    # Get the best move
    best_move = mcts.get_best_move()
    print(f"Best move: {best_move}")

if __name__ == "__main__":
    main()