# internal libraries
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
        input_layer_architecture: dict[str, any],
        hidden_layer_architectures: list[dict[str, any]],
        criterion_config: str,
        optimizer_architecture: dict[str, any]
    ):
        self.device = device
        self.device_type = device_type
        self.board_size = board_size
        self.input_layer_architecture = input_layer_architecture
        self.hidden_layer_architectures = hidden_layer_architectures
        self.criterion_config = criterion_config
        self.optimizer_architecture = optimizer_architecture
        self.model = None
    
    def initialize_model(self):
        self.model = Model(
            self.board_size,
            self.input_layer_architecture,
            self.hidden_layer_architectures,
            self.criterion_config,
            self.optimizer_architecture
        )
        self.model.eval()

    def predict(self, legal_actions: list[tuple[int, int]], state: tuple[list[list[int]], int]):
        data = prepare_data(state)
        data = np.asarray([data])
        board_width = len(state[0])
        tensor_data = torch.tensor(data, dtype=torch.float32)
        prediction_output: torch.Tensor = self.model(tensor_data)
        probability_distribution = []
        for action in legal_actions:
            action_index = action_to_index(action, board_width)
            probability_distribution.append(prediction_output[0, action_index].item())
        probability_distribution = normalize_array(probability_distribution)
        return probability_distribution
    
    def train(
        self,
        batches: tuple[tuple[list[tuple[list[list[int]], int]], list[list[float]]]]
    ):
        train_batch, validate_batch = batches
        train_loader = self.convert_batch_to_dataset(train_batch)
        validation_loader = self.convert_batch_to_dataset(validate_batch)
        self.model.train_neural_network(self.device, self.device_type, train_loader, validation_loader)
    
    def convert_batch_to_dataset(self, batch: tuple[list[tuple[list[list[int]], int]], list[list[float]]]):
        if (batch is None):
            return None
        states, visit_distributions = batch
        data = []
        labels = []
        for state in states:
            data.append(prepare_data(state))
        for visit_distribution in visit_distributions:
            labels.append(prepare_labels(visit_distribution))
        dataset_loader = convert_dataset_to_tensors(self.device_type, np.asarray(data), np.asarray(labels, dtype=np.int64))
        return dataset_loader
    
    def save(self, directory_path: str, iteration: int):
        torch.save(self.model.state_dict(), f"{directory_path}/model-{iteration}.pt")
