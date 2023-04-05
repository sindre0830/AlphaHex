# internal libraries
from functionality import (
    prepare_data
)
from model import Model
# external libraries
import random


class ANET():
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.model = None
    
    def initialize_model(self):
        self.model = Model(self.board_size)

    def predict(self, legal_actions: list[tuple[int, int]], state: tuple[list[list[int]], int]):
        board, player = state
        action_values = []
        for _ in legal_actions:
            action_values.append(random.random())
        return action_values
    
    def train(self, batch: tuple[list[tuple[list[list[int]], int]], list[list[float]]]):
        states, visit_distributions = batch
        prepare_data(states[4])
    
    def save(self, index: int):
        pass
