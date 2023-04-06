# internal libraries
from constants import (
    CPU_DEVICE,
    BATCH_SIZE
)
# external libraries
import json
import math
import numpy as np
import torch
import torch.utils.data
import multiprocessing


def parse_arguments(args: list[str]):
    error = None
    cmd = None
    cmd_args = None
    args_size = len(args)
    if (args_size <= 0):
        error = "No arguments given"
        return (error, cmd, cmd_args)
    if (args_size > 2):
        error = "Too many arguments given, require between 1 and 2"
        return (error, cmd, cmd_args)
    cmd = args[0]
    if (args_size == 2):
        cmd_args = args[1]
    return (error, cmd, cmd_args)


def print_commands():
    msg = "\nList of commands:\n"
    msg += "\t'--help' or '-h': Shows this information\n"
    msg += "\t'--alphahex' or '-ah': Starts the alphahex program, takes parameters from 'configuration.json' file\n"
    msg += "\t'--tournament' or '-t': Starts the tournament program, requries a second argument with the directory name in 'data/'\n"
    msg += "\t'--config' or '-c': Prints current configuration\n"
    print(msg)


def parse_json(directory_path: str = "", file_name: str = "configuration") -> dict[str, any]:
    with open(directory_path + file_name + ".json", "r") as file:
        return json.load(file)


def print_json(name: str, data: dict[str, any]):
    print("\n" + name + ": " + json.dumps(data, indent=4, sort_keys=True) + "\n")


def action_from_visit_distribution(visit_distribution: list[float], board_size: int) -> tuple[int, int]:
    value = max(visit_distribution)
    index = visit_distribution.index(value)
    return index_to_action(index, board_size)


def action_to_index(action: tuple[int, int], width: int) -> int:
    row, column = action
    return (row * width) + column


def index_to_action(index: int, width: int) -> tuple[int, int]:
    row = math.floor(index / width)
    column = index - (row * width)
    return (row, column)


def opposite_player(current_player: int) -> int:
    return 2 if current_player == 1 else 1


def prepare_data(state: tuple[list[list[int]], int]) -> np.ndarray:
    board, player = state
    board_width = len(board)
    data: np.ndarray = np.zeros(shape=(3, board_width, board_width))
    for row in range(board_width):
        for column in range(board_width):
            data[board[row][column]][row][column] = 1
    if player != 1:
        data[[1, 2]] = data[[2, 1]]
    return data


def prepare_labels(visit_distribution: list[float]) -> np.ndarray:
    label = np.zeros(shape=len(visit_distribution), dtype=np.float32)
    for i in range(len(visit_distribution)):
        label[i] = visit_distribution[i]
    return label


def convert_dataset_to_tensors(device_type: str, data: np.ndarray, labels: np.ndarray):
    """
    Converts dataset to a PyTorch tensor dataset.
    """
    # reshape data by adding channels
    data = np.expand_dims(data, axis=1).astype('float32')
    # convert to tensors
    data = torch.tensor(data)
    labels = torch.tensor(labels)
    # convert to dataset
    dataset = torch.utils.data.TensorDataset(data, labels)
    # convert to data loader
    pin_memory = False
    workers = 0
    # branch if device is set to CPU and set parameters accordingly
    if device_type is CPU_DEVICE:
        pin_memory = True
        workers = multiprocessing.cpu_count()
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=pin_memory, num_workers=workers)
