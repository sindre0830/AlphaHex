# internal libraries
from constants import (
    DATA_PATH,
    PLAYER_1,
    PLAYER_2
)
from functionality.data import (
    parse_json
)
from anet import ANET
from state_manager import StateManager
# external libraries
import os
import glob
import torch
import itertools


class TOPP:
    def __init__(self, device: torch.cuda.device, device_type: str, alphahex_directory_names: str):
        models: list[ANET] = []
        self.model_iterations: list[str] = []
        self.grid_size = None
        # remove duplicates
        alphahex_directory_names = list(dict.fromkeys(alphahex_directory_names))
        for alphahex_index in range(len(alphahex_directory_names)):
            working_directory_path = f"{DATA_PATH}/{alphahex_directory_names[alphahex_index]}"
            # load config
            configuration = parse_json(working_directory_path + "/", "configuration")
            if self.grid_size is None:
                self.grid_size = configuration["game_board_size"]
            elif self.grid_size != configuration["game_board_size"]:
                raise Exception("The grid size needs to be the same for all models in TOPP.")
            # load models
            model_file_paths = glob.glob(f"{working_directory_path}/*.pt")
            for model_file_path in model_file_paths:
                # get model
                anet = ANET(
                    device,
                    device_type,
                    self.grid_size,
                    configuration["epochs"],
                    configuration["input_layer"],
                    configuration["hidden_layers"],
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
                if score == PLAYER_1:
                    self.total_score[iteration_1]["Total"] += 1
                    self.total_score[iteration_1]["Player 1"] += 1
                elif score == PLAYER_2:
                    self.total_score[iteration_2]["Total"] += 1
                    self.total_score[iteration_2]["Player 2"] += 1
    
    def print_score(self):
        iterations = sorted(self.model_iterations)
        for iteration in iterations:
            print(f"Model saved at {iteration}: {self.total_score[iteration]}")
    
    def match(self, model_1: ANET, model_2: ANET):
        score: list[int] = []
        for _ in range(25):
            state_manager = StateManager()
            state_manager.initialize_state(self.grid_size)
            while not state_manager.terminal():
                legal_actions = state_manager.legal_actions()
                state = (state_manager.grid, state_manager.player)
                if state_manager.player == PLAYER_1:
                    probability_distribution = model_1.predict(legal_actions, state)
                else:
                    probability_distribution = model_2.predict(legal_actions, state)
                state_manager.apply_action_from_distribution(probability_distribution, deterministic=False)
            score.append(state_manager.determine_winner())
        return score
