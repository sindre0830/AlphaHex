# internal libraries
from constants import (
    CPU_DEVICE,
    BATCH_SIZE,
    DATA_PATH,
    INPUT_CHANNELS
)
from game_manager.hex import (
    get_legal_actions
)
import strategies
# external libraries
import json
import math
import numpy as np
import torch
import torch.utils.data
import multiprocessing
import tqdm
from typing import Iterator
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon
from matplotlib.collections import LineCollection


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


def store_json(data: dict[str, any], directory_path: str = "", file_name: str = "configuration"):
    with open(directory_path + file_name + ".json", "w") as file:
        json.dump(data, file)


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


def get_progressbar(iter: torch.utils.data.DataLoader, epoch: int, epochs: int):
    """
    Generates progressbar for iterable used in model training.
    """
    width = len(str(epochs))
    progressbar = tqdm.tqdm(
        iterable=iter,
        desc=f'                Epoch {(epoch + 1):>{width}}/{epochs}',
        ascii='░▒',
        unit=' steps',
        colour='blue'
    )
    set_progressbar_prefix(progressbar)
    return progressbar


def set_progressbar_prefix(
    progressbar: tqdm.tqdm,
    train_loss: float = 0.0,
    train_accuracy: float = 0.0,
    validation_loss: float = 0.0,
    validation_accuracy: float = 0.0
):
    """
    Set prefix in progressbar and update output.
    """
    train_loss_str = f'Train loss: {train_loss:.4f}, '
    train_accuracy_str = f'Train acc: {train_accuracy:.4f}, '
    validation_loss_str = f'Valid loss: {validation_loss:.4f}, '
    validation_accuracy_str = f'Valid acc: {validation_accuracy:.4f}'
    progressbar.set_postfix_str(train_loss_str + train_accuracy_str + validation_loss_str + validation_accuracy_str)


def normalize_array(arr: list[float]) -> list[float]:
    # divide each element by the sum
    arr_sum = sum(arr)
    arr = [elem / arr_sum for elem in arr]
    return arr


def build_hidden_layer(architecture: dict[str, any]) -> torch.nn.Sequential:
    layer_type = architecture["type"]
    match layer_type:
        case "conv":
            return torch.nn.Sequential(
                torch.nn.LazyConv2d(
                    out_channels=architecture["filters"],
                    kernel_size=architecture["kernel_size"],
                    stride=architecture["stride"],
                    padding=architecture["padding"],
                    bias=architecture["bias"]
                )
            )
        case "linear":
            return torch.nn.Sequential(
                torch.nn.LazyLinear(
                    out_features=architecture["filters"],
                    bias=architecture["bias"]
                )
            )
        case "flatten":
            return torch.nn.Sequential(
                torch.nn.Flatten()
            )
        case "max_pool":
            return torch.nn.Sequential(
                torch.nn.MaxPool2d(
                    kernel_size=architecture["kernel_size"],
                    stride=architecture["stride"]
                )
            )
        case "dropout":
            return torch.nn.Sequential(
                torch.nn.Dropout(
                    p=architecture["p"]
                )
            )
        case "batch_norm_2d":
            return torch.nn.Sequential(
                torch.nn.LazyBatchNorm2d()
            )
        case "batch_norm_1d":
            return torch.nn.Sequential(
                torch.nn.LazyBatchNorm1d()
            )
        case "relu":
            return torch.nn.Sequential(
                torch.nn.ReLU()
            )
        case _:
            return torch.nn.Sequential()


def build_criterion(criterion_config: str) -> torch.nn.modules.loss._Loss:
    match criterion_config:
        case "cross_entropy_loss":
            return torch.nn.CrossEntropyLoss()
        case "kl_divergence":
            return torch.nn.KLDivLoss(reduction="batchmean")
        case _:
            return torch.nn.CrossEntropyLoss()


def build_optimizer(parameters: Iterator[torch.nn.Parameter], architecture: dict[str, any]) -> torch.optim.Optimizer:
    optimizer_type = architecture["type"]
    match optimizer_type:
        case "adagrad":
            return torch.optim.Adagrad(
                parameters,
                lr=architecture["lr"]
            )
        case "sgd":
            return torch.optim.SGD(
                parameters,
                lr=architecture["lr"]
            )
        case "rms_prop":
            return torch.optim.RMSprop(
                parameters,
                lr=architecture["lr"]
            )
        case "adam":
            return torch.optim.Adam(
                parameters,
                lr=architecture["lr"]
            )
        case _:
            return torch.optim.Adam(parameters, lr=0.001)


def animate_game(save_directory_name: str, board_history: list[list[list[int]]], iteration: int):
    def animate(i):
        # clear previous canvas
        ax.cla()
        # draw polygons
        for row in range(board_size):
            for col in range(board_size):
                x, y = col + row, -col + row
                patch = RegularPolygon((x / 1.1, y / 1.9), numVertices=6, radius=0.6, orientation=np.pi / 2, facecolor=player_colors[board_history[i][row][col]], edgecolor="black", linewidth=1)
                ax.add_patch(patch)
        # draw lines
        offset = 0.8
        edge_states = [
            ((0 - offset, 0 - offset), (0 - offset, board_size - 1 + offset), 1),
            ((0 - offset, board_size - 1 + offset), (board_size - 1 + offset, board_size - 1 + offset), 2),
            ((board_size - 1 + offset, board_size - 1 + offset), (board_size - 1 + offset, 0 - offset), 1),
            ((board_size - 1 + offset, 0 - offset), (0 - offset, 0 - offset), 2)
        ]
        lines = []
        line_colors = []
        for edge_state in edge_states:
            (row_1, col_1), (row_2, col_2), player = edge_state
            x_1, y_1 = col_1 + row_1, -col_1 + row_1
            x_2, y_2 = col_2 + row_2, -col_2 + row_2
            lines.append([(x_1 / 1.1, y_1 / 1.9), (x_2 / 1.1, y_2 / 1.9)])
            line_colors.append(player_colors[player])
        ax.add_collection(LineCollection(lines, colors=line_colors, linewidths=4))
        # set size and apspect ratio of canvas
        ax.set_box_aspect(1)
        size = -0.5
        ax.set_xlim(-1 + size, 2 * board_size - 1 - size)
        ax.set_ylim(-board_size + size, board_size - size)
        ax.axis("off")
        return [ax]
    # duplicate last state
    last_element = board_history[-1]
    duplicates = [last_element] * 4
    board_history = board_history + duplicates
    player_colors = {-1: "green", 0: "none", 1: "red", 2: "blue"}
    # create gif
    fig, ax = plt.subplots(figsize=(10, 10))
    board_size = len(board_history[0])
    ani = animation.FuncAnimation(fig, animate, frames=len(board_history), interval=500)
    # save gif
    ani.save(f"{DATA_PATH}/{save_directory_name}/visualization_{iteration}.gif", writer="pillow")
