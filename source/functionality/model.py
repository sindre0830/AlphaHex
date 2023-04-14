# external libraries
import torch
import torch.utils.data
import tqdm
from typing import Iterator


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
