# internal libraries
from functionality import (
    parse_json
)
from rbuf import RBUF
from anet import ANET


class AlphaHex:
    def __init__(self, working_directory_path = ""):
        self.working_directory_path = working_directory_path
        self.rbuf = RBUF()
        self.anet = ANET()
        # load coniguration
        configuration = parse_json(directory_path=self.working_directory_path, file_name="configuration")
        self.save_interval = configuration["save_interval"]
        self.actual_games_size = configuration["actual_games_size"]
        self.game_board_size = configuration["game_board_size"]
        self.search_games_size = configuration["search_games_size"]

    def run(self):
        self.rbuf.clear()
        self.anet.initialize_model()
        for actual_game in range(self.actual_games_size):
            pass
