# internal libraries
from constants import (
    DATA_PATH
)
from functionality.data import (
    parse_json
)
from anet import ANET
from game_manager import GameManager
# external libraries
import os
import glob
import torch
import itertools
import random


class TOPP:
    def __init__(self, device: torch.cuda.device, device_type: str, alphahex_directory_names: str):
        models: list[ANET] = []
        self.model_iterations: list[str] = []
        self.board_size = None
        # remove duplicates
        alphahex_directory_names = list(dict.fromkeys(alphahex_directory_names))
        for alphahex_index in range(len(alphahex_directory_names)):
            working_directory_path = f"{DATA_PATH}/{alphahex_directory_names[alphahex_index]}"
            # load config
            configuration = parse_json(working_directory_path + "/", "configuration")
            if self.board_size is None:
                self.board_size = configuration["game_board_size"]
            elif self.board_size != configuration["game_board_size"]:
                raise Exception("The board size needs to be the same for all models.")
            # load models
            model_file_paths = glob.glob(f"{working_directory_path}/*.pt")
            for model_file_path in model_file_paths:
                # get model
                anet = ANET(
                    device,
                    device_type,
                    self.board_size,
                    configuration["epochs"],
                    configuration["input_layer"],
                    configuration["hidden_layers"],
                    configuration["criterion"],
                    configuration["optimizer"]
                )
                anet.initialize_model(saved_model_path=model_file_path)
                models.append(anet)
                # get model iteration
                filename = os.path.basename(model_file_path)
                name, _ = os.path.splitext(filename)
                self.model_iterations.append(f"{alphahex_directory_names[alphahex_index]}_{name.split('-')[-1]}")
        # sort and store models and their iteration index together
        self.models = list(zip(models, self.model_iterations))
        # results
        self.total_score: dict[str, dict[int, int]] = {}
        for (_, iteration) in self.models:
            self.total_score[iteration] = {}
            self.total_score[iteration]["Total"] = 0
            self.total_score[iteration]["Player 1"] = 0
            self.total_score[iteration]["Player 2"] = 0

    def run(self):
        # prepare all possible pairings
        pairings: list[tuple[tuple[ANET, str], tuple[ANET, str]]] = []
        for pair in itertools.combinations(self.models, 2):
            pairings.append(pair)
        opposite_pairings: list[tuple[tuple[ANET, str], tuple[ANET, str]]] = []
        for (model_1, model_2) in pairings:
            opposite_pairings.append((model_2, model_1))
        pairings += opposite_pairings
        # send them to match
        for ((model_1, iteration_1), (model_2, iteration_2)) in pairings:
            scores = self.match(model_1, model_2)
            for score in scores:
                if score == 1:
                    self.total_score[iteration_1]["Total"] += 1
                    self.total_score[iteration_1]["Player 1"] += 1
                elif score == 2:
                    self.total_score[iteration_2]["Total"] += 1
                    self.total_score[iteration_2]["Player 2"] += 1
    
    def print_score(self):
        iterations = sorted(self.model_iterations)
        for iteration in iterations:
            print(f"Model saved at {iteration}: {self.total_score[iteration]}")
    
    def match(self, model_1: ANET, model_2: ANET):
        score: list[int] = []
        for _ in range(25):
            game_manager = GameManager(self.board_size)
            game_manager.set_state(board=game_manager.empty_board())
            while not game_manager.terminal():
                actions = game_manager.legal_actions()
                state = (game_manager.board, game_manager.player)
                if game_manager.player == 1:
                    action_values = model_1.predict(actions, state)
                else:
                    action_values = model_2.predict(actions, state)
                action = random.choices(population=actions, weights=action_values, k=1)[0]
                game_manager.play_move(action)
            score.append(game_manager.get_winner())
        return score
