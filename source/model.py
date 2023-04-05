# external libraries
import torch
import torch.utils.data
import numpy as np


class Model(torch.nn.Module):
    def __init__(
        self,
        board_size: int,
        input_filter=32,
        filters=[64, 64],
        input_kernel_size = (3, 3),
        kernel_sizes=[(2, 2), (1, 1)],
        dense_shapes=[15, 30],
        activation=torch.nn.ReLU,
        optimizer='SGD',
        lossfunction='categorical_crossentropy'
    ):
        super().__init__()
        self.input_layer: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=input_filter,
                kernel_size=input_kernel_size
            ),
            activation()
        )
        self.conv_layers: list[torch.nn.Sequential] = []
        self.flatten_layer: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Flatten()
        )
        self.dense_layers: list[torch.nn.Sequential] = []
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
