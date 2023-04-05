# external libraries
import random


class ANET():
    def __init__(self):
        pass
    
    def initialize_model(self):
        pass

    def predict(self, legal_actions: list[tuple[int, int]], state: tuple[list[list[int]], int]):
        board, player = state
        action_values = []
        for _ in legal_actions:
            action_values.append(random.random())
        return action_values
    
    def train(self, batch: tuple[list[tuple[list[list[int]], int]], list[list[float]]]):
        states, visit_distributions = batch
    
    def save(self, index: int):
        pass
