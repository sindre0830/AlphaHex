# internal libraries
from constants import (
    CPU_DEVICE,
    BATCH_SIZE
)
# external libraries
import json
import numpy as np
import torch
import torch.utils.data
import multiprocessing
import math


def parse_json(directory_path: str = "", file_name: str = "configuration") -> dict[str, any]:
    with open(directory_path + file_name + ".json", "r") as file:
        return json.load(file)


def store_json(data: dict[str, any], directory_path: str = "", file_name: str = "configuration"):
    with open(directory_path + file_name + ".json", "w") as file:
        json.dump(data, file)


def print_json(name: str, data: dict[str, any]):
    print("\n" + name + ": " + json.dumps(data, indent=4, sort_keys=True) + "\n")


def convert_dataset_to_tensors(device_type: str, data: np.ndarray, labels: np.ndarray):
    """
    Converts dataset to a PyTorch tensor dataset.
    """
    # convert to tensors
    data = torch.tensor(data)
    labels = torch.tensor(labels)
    # convert to dataset
    dataset = torch.utils.data.TensorDataset(data, labels)
    # convert to data loader
    pin_memory = False
    workers = 0
    persistent_workers = False
    # branch if device is set to CPU and set parameters accordingly
    if device_type is CPU_DEVICE:
        pin_memory = True
        workers = multiprocessing.cpu_count()
        persistent_workers = True
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    return dataset_loader


def normalize_array(arr: np.ndarray) -> np.ndarray:
    return np.true_divide(arr, np.sum(arr))


def action_to_index(action: tuple[int, int], width: int) -> int:
    row, column = action
    return (row * width) + column


def index_to_action(index: int, width: int) -> tuple[int, int]:
    row = math.floor(index / width)
    column = index - (row * width)
    return (row, column)
