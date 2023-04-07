# internal libraries
from constants import (
    GPU_DEVICE,
    EPOCHS,
    BATCH_SIZE
)
from functionality import (
    get_progressbar,
    set_progressbar_prefix
)
# external libraries
import torch
import torch.utils.data


class Model(torch.nn.Module):
    def __init__(
        self,
        board_size: int,
        input_filter=32,
        filters=[64, 64],
        input_kernel_size = (3, 3),
        kernel_sizes=[(2, 2), (1, 1)],
        dense_shapes=[16, 32],
        activation=torch.nn.ReLU
    ):
        super().__init__()
        # define input layer
        self.input_layer: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=input_filter,
                kernel_size=input_kernel_size
            ),
            activation()
        )
        # define conv layers
        self.conv_layers: list[torch.nn.Sequential] = []
        for (filter, kernel_size) in tuple(zip(filters, kernel_sizes)):
            self.conv_layers.append(
                torch.nn.Sequential(
                    torch.nn.LazyConv2d(
                        out_channels=filter,
                        kernel_size=kernel_size
                    ),
                    activation()
                )
            )
        # define flatten layer
        self.flatten_layer = torch.nn.Flatten()
        # define dense layers
        self.dense_layers: list[torch.nn.Sequential] = []
        for dense_shape in dense_shapes:
            self.dense_layers.append(
                torch.nn.Sequential(
                    torch.nn.LazyLinear(
                        out_features=dense_shape
                    ),
                    activation()
                )
            )
        # define output layer
        self.output_layer: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.LazyLinear(out_features=(board_size * board_size)),
            torch.nn.Softmax(dim=1)
        )
    
    # Defines model layout.
    def forward(self, x):
        # input layer
        x = self.input_layer(x)
        # conv2d layer
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        # flatten layer
        x = self.flatten_layer(x)
        # linear layer
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        # output layer
        x = self.output_layer(x)
        return x
    
    def train_neural_network(
        self,
        device: torch.cuda.device,
        device_type: str,
        train_loader: torch.utils.data.DataLoader
    ):
        # branch if the device is set to GPU and send the model to the device
        if device_type is GPU_DEVICE:
            self.cuda(device)
        # set optimizer and criterion
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        TRAIN_SIZE = len(train_loader.dataset)
        # loop through each epoch
        for epoch in range(EPOCHS):
            correct = 0.0
            running_loss = 0.0
            total_loss = 0.0
            # define the progressbar
            progressbar = get_progressbar(train_loader, epoch, EPOCHS)
            # set model to training mode
            self.train()
            # loop through the dataset
            for i, (data, labels) in enumerate(progressbar):
                # send dataset to device
                data: torch.Tensor = data.to(device, non_blocking=True)
                labels: torch.Tensor = labels.to(device, non_blocking=True)
                # clear gradients
                optimizer.zero_grad()
                # get results
                output = self(data)
                # compute gradients through backpropagation
                loss: torch.Tensor = criterion(output, labels)
                loss.backward()
                # apply gradients
                optimizer.step()
                # calculate running loss
                running_loss += loss.item()
                total_loss += loss.item()
                # calculate accuracy
                output = torch.argmax(output, dim=1)
                correct += (output == labels).float().sum()
                # branch if iteration is on the last step and update information with current values
                if i >= (TRAIN_SIZE / BATCH_SIZE) - 1:
                    train_loss = total_loss / TRAIN_SIZE
                    train_accuracy = correct / TRAIN_SIZE
                    set_progressbar_prefix(progressbar, train_loss, train_accuracy)
                # branch if batch size is reached and update information with current values
                elif i % BATCH_SIZE == (BATCH_SIZE - 1):
                    train_loss = running_loss / (TRAIN_SIZE / BATCH_SIZE)
                    train_accuracy = correct / TRAIN_SIZE
                    set_progressbar_prefix(progressbar, train_loss, train_accuracy)
                    running_loss = 0.0
            # set model to training mode
            self.eval()
