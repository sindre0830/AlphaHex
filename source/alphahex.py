# internal libraries
from functionality import (
    parse_json
)
from rbuf import RBUF
from anet import ANET
from game_manager.hex import Hex
from mct import MCT


class AlphaHex:
    def __init__(self, working_directory_path = ""):
        self.working_directory_path = working_directory_path
        # load coniguration
        configuration = parse_json(directory_path=self.working_directory_path, file_name="configuration")
        self.save_interval: int = configuration["save_interval"]
        self.actual_games_size: int = configuration["actual_games_size"]
        self.game_board_size: int = configuration["game_board_size"]
        self.search_games_size: int = configuration["search_games_size"]
        # init objects
        self.rbuf = RBUF()
        self.anet = ANET()
        self.game_manager = Hex(self.game_board_size)
        self.mct = MCT()

    def run(self):
        self.rbuf.clear()
        self.anet.initialize_model()
        for actual_game in range(self.actual_games_size):
            print(f"Actual game {(actual_game + 1):>{len(str(self.actual_games_size))}}/{self.actual_games_size}")
            self.game_manager.initialize_empty_board()
            self.mct.initialize_root_node(self.game_manager.board)
            while not self.game_manager.terminal(self.game_manager.board):
                self.mct.set_game_board_from_root()
                for search_game in range(self.search_games_size):
                    pass
