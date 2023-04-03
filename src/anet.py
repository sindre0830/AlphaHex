import torch
import numpy as np
import random
from model import (
    Model
)


class ANET():
    def __init__(
        self,
        board_size: int,
        input_filter=32,
        filters=[64, 64],
        input_kernel_size=(3, 3),
        kernel_sizes=[(2, 2), (1, 1)],
        dense_shape=[15, 30],
        activation=torch.nn.ReLU,
        optimizer='SGD',
        lossfunction='categorical_crossentropy'
    ):
        self.model = Model(board_size, input_filter, filters, input_kernel_size, kernel_sizes, dense_shape, activation, optimizer, lossfunction)

    def predict(self, board, player, legal_actions, stochastic=True):
        tensor = self.convert_data_to_tensor(board, player)
        
        output = self.model(tensor)

        action_values = []
        for action in legal_actions:
            action_idx = self.action_to_index(action)
            action_values.append(output[0, action_idx].item())  # Use two indices to access the element
        return action_values

    def action_to_index(self, action):
        row, column = action
        index = 2 * row + (column - 1)
        return index

    def train(self, states, targets, learning_rate):
        pass

    def save_model(self, file_path):
        pass

    def convert_data_to_tensor(self, data, player):
        data = np.array(data)
        data = self.one_hot_encode(data, player)
        data = np.moveaxis(data, -1, 0)
        tensor = torch.tensor(data, dtype=torch.float32)
        return tensor

    def one_hot_encode(self, board, player):
        board_size = len(board[0])
        p1_board = np.where(board == 1, 1, 0)
        p2_board = np.where(board == 2, 1, 0)
        
        ohe = np.zeros(shape=(board_size, board_size, 2))        
        for i in range(board_size):
            for j in  range(board_size):
                if player == 1:
                    ohe[i, j] = [p1_board[i, j], p2_board[i, j]]
                else:
                    ohe[i, j] = [p2_board.T[i, j], p1_board.T[i, j]]

        return ohe
