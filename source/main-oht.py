from constants import (
    SEED,
    DATA_PATH,
    GPU_DEVICE,
    CPU_DEVICE
)

# Import and initialize your own actor 
import sys
import torch
import warnings
from time import time

# ignore warnings, this was added due to PyTorch LazyLayers spamming warnings
warnings.filterwarnings('ignore')
# get a device to run on
device_type = GPU_DEVICE if torch.cuda.is_available() else CPU_DEVICE
device = torch.device(device_type)
# set pytorch seed
torch.manual_seed(SEED)


from state_manager import StateManager
from anet import ANET
import numpy as np
from functionality.data import (
    parse_json,
    index_to_action
)
MODEL_NAME = "oht"
MODEL_ITERATION = 10
working_directory_path = f"../source/data/" + MODEL_NAME
# load config
configuration = parse_json(working_directory_path + "/", "configuration")

anet = ANET(
    device,
    device_type,
    configuration["grid_size"],
    configuration["max_epochs"],
    configuration["input_layer"],
    configuration["hidden_layers"],
    configuration["optimizer"],
    configuration["features"],
    configuration["criterion"]
)
anet.initialize_model(saved_model_path=f"{working_directory_path}/model-{MODEL_ITERATION}.pt")

# Import and override the `handle_get_action` hook in ActorClient
from ActorClient import ActorClient 
class MyClient(ActorClient):
    def handle_get_action(self, state):
        state_manager = StateManager()
        state_manager.initialize_state(configuration["grid_size"])
        state_manager.player = 2 if state[0] == 1 else 1
        i = 1
        for row in range(len(state_manager.grid)):
            for col in range(len(state_manager.grid)):
                if state[i] != 0:
                    state_manager.grid[row][col] = 2 if state[i] == 1 else 1
                i += 1

        state = (state_manager.grid, state_manager.player)
        probability_distribution = anet.predict(state, filter_actions=state_manager.illegal_actions())
        row, col = index_to_action(np.argmax(probability_distribution), state_manager.grid_size)
        return int(row), int(col)
        
# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient()
    client.run()
