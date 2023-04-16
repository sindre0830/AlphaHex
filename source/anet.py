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
        epochs: int,
        input_layer_architecture: dict[str, any],
        hidden_layer_architectures: list[dict[str, any]],
        optimizer_architecture: dict[str, any]
    ):
        self.device = device
        self.device_type = device_type
        self.board_size = board_size
        self.epochs = epochs
        self.input_layer_architecture = input_layer_architecture
        self.hidden_layer_architectures = hidden_layer_architectures
        self.optimizer_architecture = optimizer_architecture
        self.model = None
        self.prediction_cache = {}
    
    def initialize_model(self, saved_model_path: str = None, save_directory_name: str = None):
        self.model = Model(
            self.device,
            self.device_type,
            self.board_size,
            self.epochs,
            self.input_layer_architecture,
            self.hidden_layer_architectures,
            self.optimizer_architecture
        )
        if saved_model_path is not None:
            self.model.load(saved_model_path)
        if save_directory_name is not None:
            self.save(directory_path=DATA_PATH + "/" + save_directory_name, iteration=0)
        self.model.eval()

    def predict(self, legal_actions: list[tuple[int, int]], state: tuple[np.ndarray, int]):
        if len(legal_actions) == 1:
            return [1]
        key = (state[0].tobytes(), state[1])
        # branch if prediction is cached and return cached value
        if key in self.prediction_cache:
            return self.prediction_cache[key]
        # get prediction
        self.model.evaluate_mode()
        data = prepare_data(state)
        data = np.asarray([data])
        board_width = len(state[0])
        tensor_data = torch.tensor(data, dtype=torch.float32)
        prediction_output: torch.Tensor = self.model(tensor_data)
        # convert from logarithmic probability to normal probability
        prediction_output = prediction_output.exp()
        probability_distribution = []
        for action in legal_actions:
            action_index = action_to_index(action, board_width)
            probability_distribution.append(prediction_output[0, action_index].item())
        probability_distribution = normalize_array(probability_distribution)
        # cache result
        self.prediction_cache[key] = probability_distribution
        return probability_distribution
    
    def train(
        self,
        batches: tuple[tuple[tuple[np.ndarray, int], list[np.ndarray]], tuple[tuple[np.ndarray, int], list[np.ndarray]]]
    ):
        self.prediction_cache.clear()
        train_batch, validate_batch = batches
        train_loader = self.convert_batch_to_dataset(train_batch)
        validation_loader = self.convert_batch_to_dataset(validate_batch)
        self.model.train_neural_network(train_loader, validation_loader)
    
    def convert_batch_to_dataset(self, batch: tuple[tuple[np.ndarray, int], list[np.ndarray]]):
        if (batch is None):
            return None
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
