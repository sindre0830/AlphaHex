# internal libraries
from constants import (
    DATA_PATH
)
from functionality import (
    action_to_index,
    prepare_data,
    prepare_labels,
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
        criterion_config: str,
        optimizer_architecture: dict[str, any]
    ):
        self.device = device
        self.device_type = device_type
        self.board_size = board_size
        self.epochs = epochs
        self.input_layer_architecture = input_layer_architecture
        self.hidden_layer_architectures = hidden_layer_architectures
        self.criterion_config = criterion_config
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
            self.criterion_config,
            self.optimizer_architecture
        )
        if saved_model_path is not None:
            self.model.load(saved_model_path)
        if save_directory_name is not None:
            self.save(directory_path=DATA_PATH + "/" + save_directory_name, iteration=0)
        self.model.eval()

    def predict(self, legal_actions: list[tuple[int, int]], state: tuple[list[list[int]], int, int]):
        if len(legal_actions) == 1:
            return [1]
        key = (tuple(action for action in legal_actions), (tuple(tuple(row) for row in state[0]), state[1:]))
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
        prediction_output = torch.nn.functional.softmax(prediction_output, dim=1)
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
        batches: tuple[tuple[list[tuple[list[list[int]], int]], list[list[float]]]]
    ):
        self.prediction_cache.clear()
        train_batch, validate_batch = batches
        train_loader = self.convert_batch_to_dataset(train_batch)
        validation_loader = self.convert_batch_to_dataset(validate_batch)
        self.model.train_neural_network(train_loader, validation_loader)
    
    def convert_batch_to_dataset(self, batch: tuple[list[tuple[list[list[int]], int, int]], list[list[float]]]):
        if (batch is None):
            return None
        states, visit_distributions = batch
        data = []
        labels = []
        for state in states:
            data.append(prepare_data(state))
        for visit_distribution in visit_distributions:
            labels.append(prepare_labels(visit_distribution) if self.criterion_config != "kl_divergence" else np.asarray(visit_distribution, dtype=np.float32))
        dataset_loader = convert_dataset_to_tensors(
            self.device_type,
            data=np.asarray(data),
            labels=np.asarray(labels, dtype=np.int64) if self.criterion_config != "kl_divergence" else np.asarray(labels)
        )
        return dataset_loader
    
    def save(self, directory_path: str, iteration: int):
        torch.save(self.model.state_dict(), f"{directory_path}/model-{iteration}.pt")
