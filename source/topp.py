# internal libraries
from constants import (
    DATA_PATH,
    PLAYER_1,
    PLAYER_2,
    TOURNAMENT_GAMES,
    TOURNAMENT_VISUALIZATION
)
from anet import ANET
from state_manager import StateManager
import functionality.data
# external libraries
import os
import glob
import torch
import itertools


class TOPP:
    def __init__(self, device: torch.cuda.device, device_type: str, alphahex_directory_names: str):
        models: list[ANET] = []
        self.model_iterations: list[str] = []
        self.directory_paths = {}
        self.grid_size = None
        # remove duplicates
        alphahex_directory_names = list(dict.fromkeys(alphahex_directory_names))
        for alphahex_index in range(len(alphahex_directory_names)):
            working_directory_path = f"{DATA_PATH}/{alphahex_directory_names[alphahex_index]}"
            # load config
            configuration = functionality.data.parse_json(working_directory_path + "/", "configuration")
            if self.grid_size is None:
                self.grid_size = configuration["grid_size"]
            elif self.grid_size != configuration["grid_size"]:
                raise Exception("The grid size needs to be the same for all models in TOPP.")
            # load models
            model_file_paths = glob.glob(f"{working_directory_path}/*.pt")
            for model_file_path in model_file_paths:
                # get model
                anet = ANET(
                    device,
                    device_type,
                    self.grid_size,
                    configuration["max_epochs"],
                    configuration["input_layer"],
                    configuration["hidden_layers"],
                    configuration["optimizer"],
                    configuration["features"],
                    configuration["criterion"]
                )
                anet.initialize_model(saved_model_path=model_file_path)
                models.append(anet)
                # get model iteration
                filename = os.path.basename(model_file_path)
                name, _ = os.path.splitext(filename)
                self.model_iterations.append(f"{alphahex_directory_names[alphahex_index]}_{name.split('-')[-1]}")
                self.directory_paths[self.model_iterations[-1]] = alphahex_directory_names[alphahex_index]
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
            scores = self.match(((model_1, iteration_1), (model_2, iteration_2)))
            for score in scores:
                if score == PLAYER_1:
                    self.total_score[iteration_1]["Total"] += 1
                    self.total_score[iteration_1]["Player 1"] += 1
                elif score == PLAYER_2:
                    self.total_score[iteration_2]["Total"] += 1
                    self.total_score[iteration_2]["Player 2"] += 1

    def print_score(self):
        sorted_scores = sorted(self.total_score.items(), key=lambda x: x[1]['Total'], reverse=True)
        for model, scores in sorted_scores:
            print(f"{model}: {scores}")

    def match(self, pairings: tuple[tuple[ANET, str], tuple[ANET, str]]):
        ((model_1, iteration_1), (model_2, iteration_2)) = pairings
        score: list[int] = []
        for i in range(TOURNAMENT_GAMES):
            state_manager = StateManager()
            state_manager.initialize_state(self.grid_size)
            while not state_manager.terminal():
                state = (state_manager.grid, state_manager.player)
                if state_manager.player == PLAYER_1:
                    probability_distribution = model_1.predict(state, filter_actions=state_manager.illegal_actions())
                else:
                    probability_distribution = model_2.predict(state, filter_actions=state_manager.illegal_actions())
                state_manager.apply_action_from_distribution(
                    probability_distribution,
                    deterministic=True,
                    greedy_epsilon=None
                )
            score.append(state_manager.determine_winner())
            if TOURNAMENT_VISUALIZATION:
                state_manager.visualize(
                    save_directory_name=f"{self.directory_paths[iteration_1]}/topp",
                    iteration=i,
                    filename=f"{iteration_1}-vs-{iteration_2}",
                    verbose=False
                )
        return score
