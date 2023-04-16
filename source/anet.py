# internal libraries
from constants import (
    DATA_PATH,
    EMPTY
)
from model import Model
import functionality.data
import functionality.strategies
# external libraries
import torch
import numpy as np


class ANET():
    def __init__(
        self,
        device: torch.cuda.device,
        device_type: str,
        grid_size: int,
        minimum_epoch_improvement: int,
        input_layer_architecture: dict[str, any],
        hidden_layer_architectures: list[dict[str, any]],
        optimizer_architecture: dict[str, any],
        feature_architectures: list[dict[str, any]]
    ):
        self.device = device
        self.device_type = device_type
        self.grid_size = grid_size
        self.minimum_epoch_improvement = minimum_epoch_improvement
        self.input_layer_architecture = input_layer_architecture
        self.hidden_layer_architectures = hidden_layer_architectures
        self.optimizer_architecture = optimizer_architecture
        self.feature_architectures = feature_architectures
        self.input_channels = len(self.feature_architectures)
        self.model = None
    
    def initialize_model(self, saved_model_path: str = None, save_directory_name: str = None):
        self.model = Model(
            self.device,
            self.device_type,
            self.grid_size,
            self.input_channels,
            self.minimum_epoch_improvement,
            self.input_layer_architecture,
            self.hidden_layer_architectures,
            self.optimizer_architecture
        )
        if saved_model_path is not None:
            self.model.load(saved_model_path)
        if save_directory_name is not None:
            self.save(directory_path=DATA_PATH + "/" + save_directory_name, iteration=0, verbose=False)
        self.model.eval()

    def predict(self, state: tuple[np.ndarray, int], filter_actions: list[tuple[int, int]]):
        # get prediction
        self.model.evaluate_mode()
        data = np.asarray([self.get_features(state)])
        tensor_data = torch.tensor(data, dtype=torch.float32)
        probability_distribution: torch.Tensor = self.model(tensor_data)
        # convert from logarithmic probability to normal probability
        probability_distribution = probability_distribution.exp()
        # convert from tensor to numpy array
        probability_distribution = probability_distribution.detach().numpy()[0]
        # set all illegal actions to 0 and normalize distribution
        for action in filter_actions:
            probability_distribution[functionality.data.action_to_index(action, width=len(state[0]))] = 0
        probability_distribution = functionality.data.normalize_array(probability_distribution)
        return probability_distribution
    
    def train(self, train_batch: tuple[tuple[np.ndarray, int], list[np.ndarray]]):
        print("\tTraining model")
        train_loader = self.convert_batch_to_dataset(train_batch)
        self.model.train_neural_network(train_loader)
    
    def convert_batch_to_dataset(self, batch: tuple[tuple[np.ndarray, int], list[np.ndarray]]):
        states, visit_distributions = batch
        data = []
        labels = []
        for state in states:
            data.append(self.get_features(state))
        for visit_distribution in visit_distributions:
            labels.append(visit_distribution)
        dataset_loader = functionality.data.convert_dataset_to_tensors(
            self.device_type,
            data=np.asarray(data),
            labels=np.asarray(labels)
        )
        return dataset_loader
    
    def get_features(self, state: tuple[np.ndarray, int]) -> np.ndarray:
        board, player = state
        opponent = 2 if player == 1 else 1
        features = np.zeros(shape=(self.input_channels, self.grid_size, self.grid_size), dtype=np.float32)
        for index, feature_architecture in enumerate(self.feature_architectures):
            features[index] = self.build_feature(feature_architecture, board, player, opponent)
        return features
    
    def build_feature(self, architecture: dict[str, any], board: np.ndarray, player: int, opponent: int) -> np.ndarray:
        feature_type = architecture["type"]
        match feature_type:
            case "onehot_encode_player":
                return np.where(board == player, 1, 0).astype(dtype=np.float32)
            case "onehot_encode_opponent":
                return np.where(board == opponent, 1, 0).astype(dtype=np.float32)
            case "onehot_encode_empty":
                return np.where(board == EMPTY, 1, 0).astype(dtype=np.float32)
            case "onehot_encode_cell":
                return np.where(board == architecture["target"], 1, 0).astype(dtype=np.float32)
            case "constant_plane":
                return np.full_like(board, fill_value=architecture["value"], dtype=np.float32)
            case "constant_plane_player":
                return np.full_like(board, fill_value=player, dtype=np.float32)
            case "constant_plane_opponent":
                return np.full_like(board, fill_value=opponent, dtype=np.float32)
            case "winning_edges":
                return functionality.strategies.winning_edges(board, player, opponent)
            case "bridge_templates":
                return functionality.strategies.bridge_templates(board, player, opponent)
            case "critical_bridge_connections":
                return functionality.strategies.critical_bridge_connections(board, player, opponent)
            case "block":
                return functionality.strategies.block(board, player, opponent)
            case "_":
                return np.full_like(board, fill_value=0, dtype=np.float32)
    
    def save(self, directory_path: str, iteration: int, verbose=True):
        if verbose:
            print("\tSaving model")
        self.model.save(directory_path, iteration)
