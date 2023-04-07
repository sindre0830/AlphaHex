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
    def __init__(self, device: torch.cuda.device, device_type: str, board_size: int):
        self.device = device
        self.device_type = device_type
        self.board_size = board_size
        self.model = None
    
    def initialize_model(self):
        self.model = Model(self.board_size)
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
    
    def train(self, batch: tuple[list[tuple[list[list[int]], int]], list[list[float]]]):
        states, visit_distributions = batch
        data = []
        labels = []
        for state in states:
            data.append(prepare_data(state))
        for visit_distribution in visit_distributions:
            labels.append(prepare_labels(visit_distribution))
        dataset_loader = convert_dataset_to_tensors(self.device_type, np.asarray(data), np.asarray(labels, dtype=np.int64))
        self.model.train_neural_network(self.device, self.device_type, dataset_loader)
    
    def save(self, index: int):
        pass
