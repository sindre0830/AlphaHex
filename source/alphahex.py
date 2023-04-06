# internal libraries
from functionality import (
    parse_json,
    action_from_visit_distribution
)
from rbuf import RBUF
from anet import ANET
from mct import MCT
from game_manager.hex import (
    Hex,
    apply_action_to_board,
    print_state
)
# external libraries
import torch


class AlphaHex:
    def __init__(self, device: torch.cuda.device, device_type: str, working_directory_path = ""):
        self.working_directory_path = working_directory_path
        self.simulated_games_count = 0
        self.game_moves_count = 0
        # load coniguration
        configuration = parse_json(directory_path=self.working_directory_path, file_name="configuration")
        self.save_interval: int = configuration["save_interval"]
        self.actual_games_size: int = configuration["actual_games_size"]
        self.game_board_size: int = configuration["game_board_size"]
        self.search_games_size: int = configuration["search_games_size"]
        self.mini_batch_size: int = configuration["mini_batch_size"]
        # init objects
        self.rbuf = RBUF()
        self.anet = ANET(device, device_type, self.game_board_size)
        self.game_manager = Hex(self.game_board_size)
        self.mct = MCT()

    def run(self):
        self.rbuf.clear()
        self.anet.initialize_model()
        for actual_game in range(self.actual_games_size):
            print(f"Actual game {(actual_game + 1):>{len(str(self.actual_games_size))}}/{self.actual_games_size}")
            self.game_manager.initialize_empty_board()
            self.mct.set_root_node(self.game_manager.board, self.game_manager.player)
            while not self.game_manager.terminal():
                self.increment_game_moves_count()
                self.mct.update_game_board(self.mct.root_node.board)
                for search_game in range(self.search_games_size):
                    print(
                        f"\tSimulated game {(search_game + 1):>{len(str(self.search_games_size))}}/{self.search_games_size}, Total simulated games: {self.increment_simulated_games_count()}, Total moves: {self.game_moves_count}",
                        end="\r",
                        flush=True
                    )
                    leaf = self.mct.tree_search()
                    self.mct.node_expansion(leaf)
                    score = self.mct.leaf_evaluation(self.anet, leaf)
                    self.mct.backpropagate(leaf, score)
                visit_distribution = self.mct.root_node.visit_distribution()
                self.rbuf.add((self.mct.root_node.board, self.mct.root_node.player), visit_distribution)
                actual_move = action_from_visit_distribution(visit_distribution, self.game_board_size)
                self.game_manager.play_move(actual_move)
                self.mct.set_root_node(self.game_manager.board, self.game_manager.player)
            self.reset_counts()
            self.anet.train(self.rbuf.get_mini_batch(self.mini_batch_size))
            if actual_game % self.save_interval == 0:
                self.anet.save(actual_game)

    def increment_simulated_games_count(self) -> int:
        self.simulated_games_count += 1
        return self.simulated_games_count

    def increment_game_moves_count(self):
        self.game_moves_count += 1
    
    def reset_counts(self):
        self.simulated_games_count = 0
        self.game_moves_count = 0
        print()
