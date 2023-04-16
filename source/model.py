# internal libraries
from constants import (
    GPU_DEVICE,
    BATCH_SIZE,
    INPUT_CHANNELS
)
from functionality.model import (
    get_progressbar,
    set_progressbar_prefix,
    build_hidden_layer,
    build_optimizer
)
# external libraries
import torch
import torch.utils.data


class Model(torch.nn.Module):
    def __init__(
        self,
        device: torch.cuda.device,
        device_type: str,
        board_size: int,
        epochs: int,
        input_layer_architecture: dict[str, any],
        hidden_layer_architectures: list[dict[str, any]],
        optimizer_architecture: dict[str, any]
    ):
        super().__init__()
        self.device = device
        self.device_type = device_type
        self.total_epochs = epochs
        self.optimizer_architecture = optimizer_architecture
        # define input layer
        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=INPUT_CHANNELS,
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
            self.hidden_layers.append(build_hidden_layer(hidden_layer_architecture))
        # define output layer
        self.output_layer = torch.nn.Sequential(
            torch.nn.LazyLinear(out_features=(board_size * board_size)),
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
    
    def train_neural_network(
        self,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader = None
    ):
        self.training_mode()
        # set optimizer and criterion
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        optimizer = build_optimizer(self.parameters(), self.optimizer_architecture)
        TRAIN_SIZE = len(train_loader.dataset)
        # loop through each epoch
        for epoch in range(self.total_epochs):
            correct = 0.0
            running_loss = 0.0
            total_loss = 0.0
            # define the progressbar
            progressbar = get_progressbar(train_loader, epoch, self.total_epochs)
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
                loss: torch.Tensor = criterion(output, labels)
                loss.backward()
                # apply gradients
                optimizer.step()
                # calculate running loss
                running_loss += loss.item()
                total_loss += loss.item()
                # convert output from logarithmic probability to normal probability
                output = output.exp()
                # calculate accuracy
                correct += (torch.argmax(output, dim=1) == torch.argmax(labels, dim=1)).float().sum()
                # branch if iteration is on the last step and update information with current values
                if i >= (TRAIN_SIZE / BATCH_SIZE) - 1:
                    validation_loss, validation_accuracy = self.validate_neural_network(criterion, validation_loader)
                    train_loss = total_loss / TRAIN_SIZE
                    train_accuracy = correct / TRAIN_SIZE
                    set_progressbar_prefix(progressbar, train_loss, train_accuracy, validation_loss, validation_accuracy)
                # branch if batch size is reached and update information with current values
                elif i % BATCH_SIZE == (BATCH_SIZE - 1):
                    train_loss = running_loss / (TRAIN_SIZE / BATCH_SIZE)
                    train_accuracy = correct / TRAIN_SIZE
                    set_progressbar_prefix(progressbar, train_loss, train_accuracy)
                    running_loss = 0.0
            # set model to training mode
            self.eval()
        # empty GPU cache
        if self.device_type is GPU_DEVICE:
            torch.cuda.empty_cache()

    def validate_neural_network(
        self,
        criterion: torch.nn.CrossEntropyLoss,
        validation_loader: torch.utils.data.DataLoader
    ):
        if validation_loader is None:
            return 0.0, 0.0
        correct = 0.0
        total_loss = 0.0
        VALIDATION_SIZE = len(validation_loader.dataset)
        # set model to evaluation mode
        self.eval()
        # loop through the validation dataset
        for _, (data, labels) in enumerate(validation_loader):
            # send validation data to device
            data: torch.Tensor = data.to(self.device, non_blocking=True)
            labels: torch.Tensor = labels.to(self.device, non_blocking=True)
            # get validation results
            output: torch.Tensor = self(data)
            # calculate training loss for this batch
            loss: torch.Tensor = criterion(output, labels)
            total_loss += loss.item()
            # convert output from logarithmic probability to normal probability
            output = output.exp()
            # calculate accuracy
            correct += (torch.argmax(output, dim=1) == torch.argmax(labels, dim=1)).float().sum()
        # set model to train mode
        self.train()
        # calculate loss and accruacy
        loss = total_loss / VALIDATION_SIZE
        accuracy = correct / VALIDATION_SIZE
        return loss, accuracy
    
    def save(self, directory_path: str, iteration: int):
        torch.save(self.state_dict(), f"{directory_path}/model-{iteration}.pt")

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
