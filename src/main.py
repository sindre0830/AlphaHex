from game_manager.hex import *
from functionality import *
from mcts import *
from anet import *


class Main:
    def __init__(self):
        print("Starting AlphaHex...")
        self.board_size = 5
        self.game_manager = Hex(self.board_size)
        self.max_games = 1000
        self.max_game_variations = 100
        self.exploration_constant = 3.0
        self.epsilon = 0.0 # How often should a random move be chosen?
        self.input_size = self.board_size * self.board_size
        self.hidden_size = 64  # The number of neurons in the hidden layer
        self.output_size = self.board_size * self.board_size  # The number of unique actions (assuming a maximum of 2 stones per pile)
        print("Ready to initialize MCTS...")
        self.anet = ANET(self.input_size, self.hidden_size, self.output_size)
        self.mcts = MCTS(self.game_manager, self.max_games, self.max_game_variations, self.anet, self.exploration_constant, 1, self.epsilon)


    def play_game(self):
        current_state = self.game_manager.initial_state
        player = 1

        while not self.game_manager.is_terminal(current_state):
            print(f"Player {player}'s turn (AI)")
            self.mcts.game_manager.print_state(current_state)

            if player == 1:
                print("Thinking...")
                self.mcts.search(current_state)
                action = self.mcts.get_best_move()
            else:
                print("Enter your move (row and column separated by a space):")
                action = tuple(map(int, input().split()))

            if not self.game_manager.is_valid_action(current_state, action):
                print("Invalid move. Please try again.")
                continue

            current_state = self.game_manager.next_state(current_state, action, player)
            player = 2 if player == 1 else 1

        # self.mcts.game_manager.print_state(current_state)
        winner = 2 if player == 1 else 1
        print(f"Player {winner} wins!")
        
        
    def run():
        # Constants and parameters
        board_size = 5
        epsilon = 0.0
        exploration_constant = 1
        I_s = 50  # Save interval for ANET parameters
        number_actual_games = 1000  # The number of actual games to play
        number_search_games = 100  # The number of search games per move
        number_game_variations = 10 # The number of game variations for ANET to run TODO: Check this
        minibatch_size = 32  # The size of the minibatch for training ANET

        # Clear Replay Buffer (RBUF)
        replay_buffer = []

        # Randomly initialize parameters (weights and biases) of ANET
        input_size = board_size * board_size
        hidden_size = 64  # The number of neurons in the hidden layer
        output_size = board_size * board_size  # The number of unique actions (assuming a maximum of 2 stones per pile)
        actor_network = ANET(input_size, hidden_size, output_size)

        # Create the game manager
        game_manager = Hex(board_size)
        
        # Main loop
        for g_a in range(number_actual_games):
            print(f"Playing actual game {g_a + 1}/{number_actual_games}")

            # Initialize the actual game board (B_a) to an empty board.
            game_manager.reset_board()

            # Initialize the Monte Carlo Tree (MCT) to a single root, which represents sinit
            mcts_search = MCTS(
                game_manager=game_manager,
                root=Node(state=game_manager.initial_state),
                max_games=number_search_games,
                max_game_variations=number_game_variations,
                anet=actor_network,
                exploration_constant = exploration_constant,
                starting_player=1,
                epsilon=epsilon
                )

            # While B_a not in a final state:
            while not game_manager.is_terminal(game_manager.board):
                # For G_s in number search games:
                mcts_search.search(game_manager.board)

                # D = distribution of visit counts in MCT along all arcs emanating from root.
                D = mcts_search.root.get_distribution()

                # Add case (root, D) to RBUF
                replay_buffer.append((game_manager.board, D))

                # Choose actual move (M_a) based on D
                M_a = mcts_search.get_best_move()

                # Perform M_a on root to produce successor state S_s
                game_manager.play_move(M_a)

                # In MCT, retain subtree rooted at S_s; discard everything else.
                mcts_search.root = mcts_search.root.children[mcts_search.root.child_actions.index(M_a)]

                # Train ANET on a random minibatch of cases from RBUF
                if len(replay_buffer) >= minibatch_size:
                    minibatch = random.sample(replay_buffer, minibatch_size)
                    actor_network.train(minibatch)

                # if G_a modulo I_s == 0:
                if (g_a + 1) % I_s == 0:
                    # Save ANET's current parameters for later use in tournament play.
                    actor_network.save_model(f'anet_params_{g_a + 1}.h5')

if __name__ == "__main__":
    main = Main()
    main.play_game()
    main.run()
