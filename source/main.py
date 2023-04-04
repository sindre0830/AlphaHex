# internal libraries
from constants import (
    GPU_DEVICE,
    CPU_DEVICE
)
from functionality import (
    parse_arguments,
    print_commands
)
# external libraries
import sys
import torch
import warnings

# ignore warnings, this was added due to PyTorch LazyLayers spamming warnings
warnings.filterwarnings('ignore')
# get a device to run on
device_type = GPU_DEVICE if torch.cuda.is_available() else CPU_DEVICE
device = torch.device(device_type)


# Main program.
def main():
    # parse arguments and branch if an error occured
    (error, cmd, cmd_args) = parse_arguments(sys.argv[1:])
    if (error != None):
        print("Error during command parsing: '" + error + "'")
        print_commands()
        return
    # compute command given
    match cmd:
        case "--help" | "-h":
            print_commands()
            return
        case "--alphahex" | "-ah":
            print("Starting alpha hex...")
            return
        case "--tournament" | "-t":
            if (cmd_args == None):
                print("Tournament requires a directory name from 'data/' of which models to run.")
                print_commands()
                return
            print("Starting tournament...")
            return
        case _:
            print("Error during command matching 'Command not found'")
            print_commands()
            return


# branch if program is run through 'python main.py'
if __name__ == "__main__":
    main()
