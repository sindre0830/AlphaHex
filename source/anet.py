# internal libraries
from functionality import (
    action_to_index,
    prepare_data
)
from model import Model
# external libraries
import torch


class ANET():
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.model = None
    
    def initialize_model(self):
        self.model = Model(self.board_size)
        self.model.eval()

    def predict(self, legal_actions: list[tuple[int, int]], state: tuple[list[list[int]], int]):
        data = prepare_data(state)
        board_width = len(state[0])
        tensor_data = torch.tensor(data, dtype=torch.float32)
        prediction_output: torch.Tensor = self.model(tensor_data)
        action_values = []
        for action in legal_actions:
            action_index = action_to_index(action, board_width)
            action_values.append(prediction_output[0, action_index].item())
        return action_values
    
    def train(self, batch: tuple[list[tuple[list[list[int]], int]], list[list[float]]]):
        states, visit_distributions = batch
    
    def save(self, index: int):
        pass
