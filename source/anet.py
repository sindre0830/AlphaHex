# internal libraries
from constants import (
    DATA_PATH
)
from functionality.game import (
    action_to_index
)
from functionality.data import (
    prepare_data,
    convert_dataset_to_tensors,
    normalize_array
)
from model import Model
# external libraries
import torch
import numpy as np


class ANET():
    def __init__(
        self,
        device: torch.cuda.device,
        device_type: str,
        board_size: int,
        minimum_epoch_improvement: int,
        input_layer_architecture: dict[str, any],
        hidden_layer_architectures: list[dict[str, any]],
        optimizer_architecture: dict[str, any]
    ):
        self.device = device
        self.device_type = device_type
        self.board_size = board_size
        self.minimum_epoch_improvement = minimum_epoch_improvement
        self.input_layer_architecture = input_layer_architecture
        self.hidden_layer_architectures = hidden_layer_architectures
        self.optimizer_architecture = optimizer_architecture
        self.model = None
    
    def initialize_model(self, saved_model_path: str = None, save_directory_name: str = None):
        self.model = Model(
            self.device,
            self.device_type,
            self.board_size,
            self.minimum_epoch_improvement,
            self.input_layer_architecture,
            self.hidden_layer_architectures,
            self.optimizer_architecture
        )
        if saved_model_path is not None:
            self.model.load(saved_model_path)
        if save_directory_name is not None:
            self.save(directory_path=DATA_PATH + "/" + save_directory_name, iteration=0)
        self.model.eval()

    def predict(self, state: tuple[np.ndarray, int], filter_actions: list[tuple[int, int]]):
        # get prediction
        self.model.evaluate_mode()
        data = np.asarray([prepare_data(state)])
        tensor_data = torch.tensor(data, dtype=torch.float32)
        probability_distribution: torch.Tensor = self.model(tensor_data)
        # convert from logarithmic probability to normal probability
        probability_distribution = probability_distribution.exp()
        # convert from tensor to numpy array
        probability_distribution = probability_distribution.detach().numpy()[0]
        # set all illegal actions to 0 and normalize distribution
        for action in filter_actions:
            probability_distribution[action_to_index(action, width=len(state[0]))] = 0
        probability_distribution = normalize_array(probability_distribution)
        return probability_distribution
    
    def train(self, train_batch: tuple[tuple[np.ndarray, int], list[np.ndarray]]):
        train_loader = self.convert_batch_to_dataset(train_batch)
        self.model.train_neural_network(train_loader)
    
    def convert_batch_to_dataset(self, batch: tuple[tuple[np.ndarray, int], list[np.ndarray]]):
        states, visit_distributions = batch
        data = []
        labels = []
        for state in states:
            data.append(prepare_data(state))
        for visit_distribution in visit_distributions:
            labels.append(visit_distribution)
        dataset_loader = convert_dataset_to_tensors(
            self.device_type,
            data=np.asarray(data),
            labels=np.asarray(labels)
        )
        return dataset_loader
    
    def save(self, directory_path: str, iteration: int):
        self.model.save(directory_path, iteration)
