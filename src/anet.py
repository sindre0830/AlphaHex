import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ANET(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, board_size):
        super(ANET, self).__init__()
        self.board_size = board_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, board_size * board_size)
        
    def forward(self, x):
        x = x.view(-1, self.board_size * self.board_size)  # Reshape the input tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, state, legal_actions):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        state_tensor = state_tensor.view(-1)  # Flatten the input tensor
        output = self.forward(state_tensor)

        action_values = []
        for action in legal_actions:
            action_idx = self.action_to_index(action)
            action_values.append(output[0, action_idx].item())  # Use two indices to access the element
        return action_values

    def action_to_index(self, action):
        pile, stones = action
        index = 2 * pile + (stones - 1)
        return index

    def train(self, states, targets, learning_rate):
        states_tensor = torch.tensor(states, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        optimizer.zero_grad()
        outputs = self.forward(states_tensor)
        loss = criterion(outputs, targets_tensor)
        loss.backward()
        optimizer.step()

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
