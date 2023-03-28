import torch
import torch.nn as nn
import torch.optim as optim


class ANET(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANET, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        print("Initiating ANET...")

        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


    def predict(self, state, legal_actions):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        state_tensor = state_tensor.view(-1)  # Flatten the input tensor
        output = self.forward(state_tensor)

        action_values = []
        for action in legal_actions:
            action_idx = self.action_to_index(action)
            action_values.append(output[action_idx].item())
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