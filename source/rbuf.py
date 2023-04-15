# internal libraries
from functionality.data import (
    normalize_array
)
# external libraries
import numpy as np


class RBUF():
    def __init__(self):
        self.data: list[tuple[np.ndarray, int]] = []
        self.labels: list[np.ndarray] = []
        self.frequency_count: dict[tuple[bytes, int, bytes], int] = {}
        self.previous_training_batch: tuple[tuple[np.ndarray, int], list[np.ndarray]] = None

    def clear(self):
        self.data.clear()
        self.labels.clear()
        self.frequency_count.clear()

    def add(self, state: tuple[np.ndarray, int], visit_distribution: np.ndarray):
        self.data.append(state)
        self.labels.append(visit_distribution)
        self.frequency_count[self.key((state, visit_distribution))] = 1

    def get_mini_batch(self, mini_batch_size) -> tuple[tuple[tuple[np.ndarray, int], list[np.ndarray]], tuple[tuple[np.ndarray, int], list[np.ndarray]]]:
        data_size = len(self.data)
        indicies = list(range(data_size))
        if (data_size <= mini_batch_size):
            train_indicies = indicies
        else:
            train_indicies = np.random.choice(
                indicies,
                size=mini_batch_size,
                replace=False,
                p=self.weights()
            )
        train_data = []
        train_labels = []
        for index in range(data_size):
            if index in train_indicies:
                train_data.append(self.data[index])
                train_labels.append(self.labels[index])
        train_batch = (train_data, train_labels)
        validation_batch = self.previous_training_batch
        self.previous_training_batch = train_batch
        self.increment_frequency_count(train_batch)
        return (train_batch, validation_batch)

    def increment_frequency_count(self, batch: tuple[list[tuple[np.ndarray, int]], list[np.ndarray]]):
        data, labels = batch
        for dataset in list(zip(data, labels)):
            self.frequency_count[self.key(dataset)] += 1

    def weights(self) -> list[float]:
       weights = [1 / self.frequency_count[self.key(dataset)] for dataset in list(zip(self.data, self.labels))]
       return normalize_array(weights)

    def key(self, dataset: tuple[tuple[np.ndarray, int], np.ndarray]):
        return (dataset[0][0].tobytes(), dataset[0][1], dataset[1].tobytes())
