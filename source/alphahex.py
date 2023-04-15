# internal libraries
from constants import (
    DATA_PATH
)
from functionality.data import (
    parse_json,
    store_json
)
from functionality.game import (
    action_from_visit_distribution,
    animate_game
)
from rbuf import RBUF
from anet import ANET
from mcts import MCTS
from game_manager import GameManager
# external libraries
import torch
from time import time
import os
from datetime import datetime


class AlphaHex:
    def __init__(self, device: torch.cuda.device, device_type: str):
        self.simulated_games_count = 0
        self.game_moves_count = 0
        # load coniguration
        self.configuration = parse_json(file_name="configuration")
        self.save_interval: int = self.configuration["save_interval"]
        self.save_visualization_interval: int = self.configuration["save_visualization_interval"]
        self.actual_games_size: int = self.configuration["actual_games_size"]
        self.game_board_size: int = self.configuration["game_board_size"]
        self.search_games_size: int = self.configuration["search_games_size"]
        self.mini_batch_size: int = self.configuration["mini_batch_size"]
        # init objects
        self.rbuf = RBUF()
        self.anet = ANET(
            device,
            device_type,
            self.game_board_size,
            self.configuration["epochs"],
            self.configuration["input_layer"],
            self.configuration["hidden_layers"],
            self.configuration["criterion"],
            self.configuration["optimizer"]
        )
        self.game_manager = GameManager(self.game_board_size)
        self.mcts = MCTS()
        # create directory to store models and configuration
        self.save_directory_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(f"{DATA_PATH}/{self.save_directory_name}", exist_ok=True)
        store_json(self.configuration, directory_path=f"{DATA_PATH}/{self.save_directory_name}/", file_name="configuration")

    def run(self):
        self.rbuf.clear()
        self.anet.initialize_model(save_directory_name=self.save_directory_name)
        for actual_game in range(self.actual_games_size):
            print(f"Actual game {(actual_game + 1):>{len(str(self.actual_games_size))}}/{self.actual_games_size}")
            time_start = time()
            self.game_manager.set_state(board=self.game_manager.empty_board())
            self.mcts.set_root_node(self.game_manager.board, self.game_manager.player, reset_turn=True)
            while not self.game_manager.terminal():
                self.increment_game_moves_count()
                self.mcts.update_game_board(self.mcts.root_node.board)
                for search_game in range(self.search_games_size):
                    print(
                        f"\tSimulated game {(search_game + 1):>{len(str(self.search_games_size))}}/{self.search_games_size}, Total simulated games: {self.increment_simulated_games_count()}, Total moves: {self.game_moves_count}",
                        end="\r",
                        flush=True
                    )
                    leaf = self.mcts.tree_search()
                    self.mcts.node_expansion(leaf)
                    score = self.mcts.leaf_evaluation(self.anet, leaf)
                    self.mcts.backpropagate(leaf, score)
                visit_distribution = self.mcts.root_node.visit_distribution()
                self.rbuf.add((self.mcts.root_node.board, self.mcts.root_node.player), visit_distribution)
                actual_move = action_from_visit_distribution(visit_distribution, self.game_board_size)
                self.game_manager.play_move(actual_move)
                self.mcts.set_root_node(self.game_manager.board, self.game_manager.player)
            self.reset_counts()
            print("\tTraining model")
            self.anet.train(self.rbuf.get_mini_batch(self.mini_batch_size))
            if (actual_game + 1) % self.save_interval == 0 or actual_game == (self.actual_games_size - 1):
                self.anet.save(directory_path=f"{DATA_PATH}/{self.save_directory_name}", iteration=(actual_game + 1))
                print("\tModel saved")
            if self.save_visualization_interval is not None and ((actual_game + 1) % self.save_visualization_interval == 0 or actual_game == (self.actual_games_size - 1) or actual_game == 0):
                animate_game(self.save_directory_name, self.mcts.game_board_history, iteration=(actual_game + 1))
                print("\tBoard visualization saved")
            time_end = time()
            print(f"\tTime elapsed: {(time_end - time_start):0.2f} seconds")

    def increment_simulated_games_count(self) -> int:
        self.simulated_games_count += 1
        return self.simulated_games_count

    def increment_game_moves_count(self):
        self.game_moves_count += 1
    
    def reset_counts(self):
        self.simulated_games_count = 0
        self.game_moves_count = 0
        print()
