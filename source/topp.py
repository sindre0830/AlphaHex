# internal libraries
from constants import (
    DATA_PATH
)
from functionality import (
    parse_json
)
from anet import ANET
from game_manager.hex import (
    Hex,
    terminal,
    get_legal_actions,
    apply_action_to_board
)
# external libraries
import os
import glob
import torch
import itertools
import random


class TOPP:
    def __init__(self, device: torch.cuda.device, device_type: str, alphahex_directory_name: str):
        self.working_directory_path = f"{DATA_PATH}/{alphahex_directory_name}"
        # load config
        self.configuration = parse_json(self.working_directory_path + "/", "configuration")
        self.board_size = self.configuration["game_board_size"]
        # load models
        model_file_paths = glob.glob(f"{self.working_directory_path}/*.pt")
        models: list[ANET] = []
        model_iterations: list[int] = []
        for model_file_path in model_file_paths:
            # get model
            anet = ANET(
                device,
                device_type,
                self.board_size,
                self.configuration["epochs"],
                self.configuration["input_layer"],
                self.configuration["hidden_layers"],
                self.configuration["criterion"],
                self.configuration["optimizer"]
            )
            anet.initialize_model(saved_model_path=model_file_path)
            models.append(anet)
            # get model iteration
            filename = os.path.basename(model_file_path)
            name, _ = os.path.splitext(filename)
            model_iterations.append(int(name.split('-')[-1]))
        # sort and store models and their iteration index together
        self.models = list(zip(models, model_iterations))
        # results
        self.total_score: dict[int, int] = {}
        for (_, iteration) in self.models:
            self.total_score[iteration] = 0

    def run(self):
        # prepare all possible pairings
        pairings: list[tuple[tuple[ANET, int], tuple[ANET, int]]] = []
        for pair in itertools.combinations(self.models, 2):
            pairings.append(pair)
        opposite_pairings: list[tuple[tuple[ANET, int], tuple[ANET, int]]] = []
        for (model_1, model_2) in pairings:
            opposite_pairings.append((model_2, model_1))
        pairings += opposite_pairings
        # send them to match
        for ((model_1, iteration_1), (model_2, iteration_2)) in pairings:
            scores = self.match(model_1, model_2)
            for score in scores:
                if score == 1:
                    self.total_score[iteration_1] += 1
                elif score == 2:
                    self.total_score[iteration_2] += 1
        print(self.total_score)
    
    def match(self, model_1: ANET, model_2: ANET):
        score: list[int] = []
        for _ in range(25):
            game_manager = Hex(self.board_size)
            game_manager.initialize_empty_board()
            while not game_manager.terminal():
                legal_actions = game_manager.get_legal_actions()
                state = (game_manager.board, game_manager.player)
                if game_manager.player == 1:
                    action_values = model_1.predict(legal_actions, state)
                else:
                    action_values = model_2.predict(legal_actions, state)
                action = random.choices(population=legal_actions, weights=action_values, k=1)[0]
                game_manager.play_move(action)
            score.append(game_manager.get_winner())
        return score
