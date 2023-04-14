# internal libraries
from constants import (
    CPU_DEVICE,
    BATCH_SIZE,
    INPUT_CHANNELS
)
from game_manager import (
    get_legal_actions
)
import functionality.strategies as strategies
from functionality.game import (
    opposite_player
)
# external libraries
import json
import numpy as np
import torch
import torch.utils.data
import multiprocessing


def parse_json(directory_path: str = "", file_name: str = "configuration") -> dict[str, any]:
    with open(directory_path + file_name + ".json", "r") as file:
        return json.load(file)


def store_json(data: dict[str, any], directory_path: str = "", file_name: str = "configuration"):
    with open(directory_path + file_name + ".json", "w") as file:
        json.dump(data, file)


def print_json(name: str, data: dict[str, any]):
    print("\n" + name + ": " + json.dumps(data, indent=4, sort_keys=True) + "\n")


def prepare_data(state: tuple[list[list[int]], int, int]) -> np.ndarray:
    board, player, turn = state
    opponent = opposite_player(player)
    board_size = len(board)
    # prepare feature maps
    data: np.ndarray = np.zeros(shape=(INPUT_CHANNELS, board_size, board_size), dtype=np.float32)
    # fill the first 3 feature maps with current board data
    for row in range(board_size):
        for column in range(board_size):
            cell = board[row][column]
            if cell == player:
                data[0][row][column] = 1
            elif cell == opponent:
                data[1][row][column] = 1
            else:
                data[2][row][column] = 1
    data[3] = 1
    data[4] = strategies.fork_actions(board, player)
    for (row, column) in get_legal_actions(board):
        data[5][row][column] = 1
    data[6] = 0
    return data


def prepare_labels(visit_distribution: list[float]) -> float:
    return visit_distribution.index(max(visit_distribution))


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


def normalize_array(arr: list[float]) -> list[float]:
    # divide each element by the sum
    arr_sum = sum(arr)
    arr = [elem / arr_sum for elem in arr]
    return arr
