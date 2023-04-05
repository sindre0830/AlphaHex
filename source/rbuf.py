# external libraries
import random


class RBUF():
    def __init__(self):
        self.states: list[tuple[list[list[int]], int]] = []
        self.visit_distributions: list[list[float]] = []

    def clear(self):
        self.states.clear()
        self.visit_distributions.clear()

    def add(self, state, visit_distribution):
        self.states.append(state)
        self.visit_distributions.append(visit_distribution)

    def get_mini_batch(self, mini_batch_size) -> tuple[list[tuple[list[list[int]], int]], list[list[float]]]:
        if (len(self.states) <= mini_batch_size):
            return (self.states, self.visit_distributions)
        else:
            return zip(*random.sample(list(zip(self.states, self.visit_distributions)), mini_batch_size))
