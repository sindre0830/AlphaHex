# internal libraries
from constants import (
    CPU_DEVICE,
    BATCH_SIZE,
    INPUT_CHANNELS
)
import functionality.strategies as strategies
from functionality.game import (
    opposite_player
)
import functionality.feature_maps as feature_maps
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


def prepare_data(state: tuple[np.ndarray, int]) -> np.ndarray:
    board, player = state
    opponent = opposite_player(player)
    board_size = len(board)
    # prepare feature maps
    data: np.ndarray = np.zeros(shape=(INPUT_CHANNELS, board_size, board_size), dtype=np.float32)
    # fill the first 3 feature maps with current board data
    data[0] = feature_maps.onehot_encode_cell(board, target=player)
    data[1] = feature_maps.onehot_encode_cell(board, target=opponent)
    data[2] = feature_maps.onehot_encode_cell(board, target=0)
    data[3] = feature_maps.constant_plane(board, value=1)
    data[4] = feature_maps.strategy(strategies.winning_edges, board, player)
    data[5] = feature_maps.strategy(strategies.bridge_templates, board, player)
    data[6] = feature_maps.strategy(strategies.critical_bridge_connections, board, player)
    data[7] = feature_maps.strategy(strategies.block, board, player)
    data[8] = feature_maps.constant_plane(board, value=0)
    return data


def prepare_labels(visit_distribution: np.ndarray) -> int:
    return np.argmax(visit_distribution)


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
    # divide each element by the sum
    arr_sum = sum(arr)
    arr = [elem / arr_sum for elem in arr]
    return arr
