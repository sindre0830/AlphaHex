# internal libraries
from constants import (
    DATA_PATH
)
from rbuf import RBUF
from anet import ANET
from mcts import MCTS
from state_manager import StateManager
import functionality.data
# external libraries
import torch
from time import time
import os
from datetime import datetime


class AlphaHex:
    def __init__(self, device: torch.cuda.device, device_type: str):
        self.simulated_games_count = 0
        # load coniguration
        self.configuration = functionality.data.parse_json(file_name="configuration")
        self.save_interval: int = self.configuration["save_interval"]
        self.save_visualization_interval: int = self.configuration["save_visualization_interval"]
        self.actual_games_size: int = self.configuration["actual_games_size"]
        self.grid_size: int = self.configuration["grid_size"]
        self.search_games_time_limit_seconds: int = self.configuration["search_games_time_limit_seconds"]
        self.mini_batch_size: int = self.configuration["mini_batch_size"]
        # init objects
        self.rbuf = RBUF()
        self.anet = ANET(
            device,
            device_type,
            self.grid_size,
            self.configuration["minimum_epoch_improvement"],
            self.configuration["input_layer"],
            self.configuration["hidden_layers"],
            self.configuration["optimizer"],
            self.configuration["features"]
        )
        self.state_manager = StateManager()
        self.mcts = MCTS()
        # create directory to store models and configuration
        self.save_directory_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(f"{DATA_PATH}/{self.save_directory_name}", exist_ok=True)
        functionality.data.store_json(
            self.configuration,
            directory_path=f"{DATA_PATH}/{self.save_directory_name}/",
            file_name="configuration"
        )

    def run(self):
        self.rbuf.clear()
        self.anet.initialize_model(save_directory_name=self.save_directory_name)
        for actual_game in range(self.actual_games_size):
            print(f"Actual game {(actual_game + 1):>{len(str(self.actual_games_size))}}/{self.actual_games_size}")
            time_start = time()
            self.state_manager.initialize_state(self.grid_size)
            self.mcts.set_root_node(self.state_manager)
            while not self.state_manager.terminal():
                search_games_time_start = time()
                while((time() - search_games_time_start) < self.search_games_time_limit_seconds):
                    leaf = self.mcts.tree_search()
                    self.mcts.node_expansion(leaf)
                    score = self.mcts.leaf_evaluation(self.anet, leaf)
                    self.mcts.backpropagate(leaf, score)
                    print(
                        f"\tTotal simulated games: {self.increment_simulated_games_count()}, Total moves: {self.state_manager.round()}",
                        end="\r",
                        flush=True
                    )
                visit_distribution = self.mcts.root_node.visit_distribution()
                self.rbuf.add(self.mcts.root_node.state, visit_distribution)
                self.state_manager.apply_action_from_distribution(visit_distribution, deterministic=True)
                self.mcts.set_root_node(self.state_manager)
            self.reset_simulated_games_count()
            self.anet.train(self.rbuf.get_mini_batch(self.mini_batch_size))
            save_model = (actual_game + 1) % self.save_interval == 0 or actual_game == (self.actual_games_size - 1)
            if save_model:
                self.anet.save(directory_path=f"{DATA_PATH}/{self.save_directory_name}", iteration=(actual_game + 1))
            visualize_state = (actual_game + 1) % self.save_visualization_interval == 0 or actual_game == (self.actual_games_size - 1) or actual_game == 0
            if visualize_state:
                self.state_manager.visualize(self.save_directory_name, iteration=(actual_game + 1))
            time_end = time()
            print(f"\tTime elapsed: {(time_end - time_start):0.2f} seconds")

    def increment_simulated_games_count(self) -> int:
        self.simulated_games_count += 1
        return self.simulated_games_count
    
    def reset_simulated_games_count(self):
        self.simulated_games_count = 0
        print()
