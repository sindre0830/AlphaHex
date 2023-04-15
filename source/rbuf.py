# external libraries
import random
import numpy as np


class RBUF():
    def __init__(self):
        self.states: list[tuple(np.ndarray, int)] = []
        self.visit_distributions: list[np.ndarray] = []

    def clear(self):
        self.states.clear()
        self.visit_distributions.clear()

    def add(self, state, visit_distribution):
        self.states.append(state)
        self.visit_distributions.append(visit_distribution)

    def get_mini_batch(self, mini_batch_size) -> tuple[tuple[tuple[np.ndarray, int], list[np.ndarray]], tuple[tuple[np.ndarray, int], list[np.ndarray]]]:
        if (len(self.states) <= mini_batch_size):
            train_batch = (self.states, self.visit_distributions)
            validation_batch = None
            return (train_batch, validation_batch)
        else:
            selected_data = random.sample(population=list(zip(self.states, self.visit_distributions)), k=mini_batch_size)
            all_data = list(zip(self.states, self.visit_distributions))
            train_batch = zip(*selected_data)
            validation_batch = zip(*[data for data in all_data if data not in selected_data])
            return (train_batch, validation_batch)
