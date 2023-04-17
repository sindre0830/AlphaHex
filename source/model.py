# internal libraries
from constants import (
    TMP_PATH,
    GPU_DEVICE,
    VERBOSE_TRAINING,
    BATCH_SIZE
)
# external libraries
import torch
import torch.utils.data
import tqdm
from typing import Iterator
import os


class Model(torch.nn.Module):
    def __init__(
        self,
        device: torch.cuda.device,
        device_type: str,
        grid_size: int,
        input_channels: int,
        max_epochs: int,
        input_layer_architecture: dict[str, any],
        hidden_layer_architectures: list[dict[str, any]],
        optimizer_architecture: dict[str, any],
        criterion_type: str
    ):
        super().__init__()
        self.device = device
        self.device_type = device_type
        self.max_epochs = max_epochs
        self.optimizer_architecture = optimizer_architecture
        self.criterion_type = criterion_type
        # create tmp folder for temporary model saving
        os.makedirs(TMP_PATH, exist_ok=True)
        # define input layer
        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=input_layer_architecture["filters"],
                kernel_size=input_layer_architecture["kernel_size"],
                stride=input_layer_architecture["stride"],
                padding=input_layer_architecture["padding"],
                bias=input_layer_architecture["bias"]
            )
        )
        # define hidden layers
        self.hidden_layers: list[torch.nn.Sequential] = []
        for hidden_layer_architecture in hidden_layer_architectures:
            self.hidden_layers.append(self.build_hidden_layer(hidden_layer_architecture))
        # define output layer
        self.output_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(out_features=(grid_size * grid_size)),
            torch.nn.LogSoftmax(dim=1)
        )
        # send model to CPU
        self.training_flag = None
        self.evaluate_mode()

    def evaluate_mode(self):
        if self.training_flag is False:
            return
        else:
            self.training_flag = False
        # send model to CPU
        self.cpu()
        for hidden_layer in self.hidden_layers:
            hidden_layer.cpu()

    def training_mode(self):
        if self.training_flag is True:
            return
        else:
            self.training_flag = True
        # branch if the device is set to GPU and send the model to the device
        if self.device_type is GPU_DEVICE:
            self.cuda(self.device)
            for hidden_layer in self.hidden_layers:
                hidden_layer.to(self.device)

    # Defines model layout.
    def forward(self, x):
        # input layer
        x = self.input_layer(x)
        # hidden layers
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        # output layer
        x = self.output_layer(x)
        return x

    def train_neural_network(self, train_loader: torch.utils.data.DataLoader):
        self.training_mode()
        # set optimizer and criterion
        criterion = self.build_criterion(self.criterion_type)
        optimizer = self.build_optimizer(self.parameters(), self.optimizer_architecture)
        TRAIN_SIZE = len(train_loader.dataset)
        # loop through each epoch
        best_loss = float("inf")
        best_accuracy = -float("inf")
        for epoch in range(self.max_epochs):
            correct = 0.0
            total_loss = 0.0
            # define the progressbar
            progressbar = self.get_progressbar(train_loader, epoch, self.max_epochs)
            # set model to training mode
            self.train()
            # loop through the dataset
            for i, (data, labels) in enumerate(progressbar):
                # send dataset to device
                data: torch.Tensor = data.to(self.device, non_blocking=True)
                labels: torch.Tensor = labels.to(self.device, non_blocking=True)
                # clear gradients
                optimizer.zero_grad()
                # get results
                output: torch.Tensor = self(data)
                # compute gradients through backpropagation
                targets = labels.argmax(dim=1) if self.criterion_type == "cross_entropy" else labels
                loss: torch.Tensor = criterion(output, targets)
                loss.backward()
                # apply gradients
                optimizer.step()
                # convert output from logarithmic probability to normal probability
                output = output.exp()
                # calculate total loss and accuracy
                total_loss += loss.item()
                correct += (torch.argmax(output, dim=1) == torch.argmax(labels, dim=1)).float().sum()
                # branch if iteration is on the last step and update information with current values
                if i >= (TRAIN_SIZE / BATCH_SIZE) - 1:
                    train_loss = total_loss / TRAIN_SIZE
                    train_accuracy = correct / TRAIN_SIZE
                    if train_loss <= best_loss:
                        best_loss = train_loss
                        best_accuracy = train_accuracy
                        # save best model
                        self.save(TMP_PATH, filename="tmp_model")
                    self.set_progressbar_prefix(progressbar, train_loss, train_accuracy, best_loss, best_accuracy)
        # load best model from training
        self.load(path=f"{TMP_PATH}/tmp_model.pt")
        # empty GPU cache
        if self.device_type is GPU_DEVICE:
            torch.cuda.empty_cache()

    def save(self, directory_path: str, filename: str):
        torch.save(self.state_dict(), f"{directory_path}/{filename}.pt")

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    def get_progressbar(self, iter: torch.utils.data.DataLoader, epoch: int, max_epochs: int) -> tqdm.tqdm:
        """
        Generates progressbar for iterable used in model training.
        """
        width = len(str(max_epochs))
        progressbar = tqdm.tqdm(
            iterable=iter,
            desc=f'                Epoch {(epoch + 1):>{width}}/{max_epochs}',
            ascii='░▒',
            unit=' steps',
            colour='blue',
            disable=(not VERBOSE_TRAINING)
        )
        self.set_progressbar_prefix(progressbar)
        return progressbar

    def set_progressbar_prefix(
        self,
        progressbar: tqdm.tqdm,
        train_loss: float = 0.0,
        train_accuracy: float = 0.0,
        best_loss: float = 0.0,
        best_accuracy: float = 0.0
    ):
        """
        Set prefix in progressbar and update output.
        """
        train_loss_str = f'loss: {train_loss:.4f}, '
        train_accuracy_str = f'acc: {train_accuracy:.4f}, '
        best_loss_str = f'best loss: {best_loss:.4f}, '
        best_accuracy_str = f'best acc: {best_accuracy:.4f}'
        progressbar.set_postfix_str(train_loss_str + train_accuracy_str + best_loss_str + best_accuracy_str)

    def build_criterion(self, criterion_type: str):
        match criterion_type:
            case "kl_div":
                return torch.nn.KLDivLoss(reduction="batchmean")
            case "cross_entropy":
                return torch.nn.CrossEntropyLoss()
            case _:
                return torch.nn.CrossEntropyLoss()

    def build_optimizer(self, parameters: Iterator[torch.nn.Parameter], architecture: dict[str, any]) -> torch.optim.Optimizer:
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

    def build_hidden_layer(self, architecture: dict[str, any]) -> torch.nn.Sequential:
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
            case "sigmoid":
                return torch.nn.Sequential(
                    torch.nn.Sigmoid()
                )
            case "tanh":
                return torch.nn.Sequential(
                    torch.nn.Tanh()
                )
            case "linear":
                return torch.nn.Sequential()
            case _:
                return torch.nn.Sequential()
