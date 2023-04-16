# internal libraries
from constants import (
    DATA_PATH,
    GPU_DEVICE,
    CPU_DEVICE
)
from alphahex import AlphaHex
from topp import TOPP
import functionality.cli
import functionality.data
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
    (error, cmd, cmd_args) = functionality.cli.parse_arguments(args=sys.argv[1:])
    if (error != None):
        print("Error during command parsing: '" + error + "'")
        functionality.cli.print_commands()
        return
    # compute command given
    match cmd:
        case "--help" | "-h":
            functionality.cli.print_commands()
            return
        case "--alphahex" | "-ah":
            print("Starting alpha hex...")
            alpha_hex = AlphaHex(device, device_type)
            alpha_hex.run()
            return
        case "--tournament" | "-t":
            if (cmd_args == None):
                print("Tournament requires a directory name from 'data/' of which models to run.")
                functionality.cli.print_commands()
                return
            print("Starting tournament...")
            topp = TOPP(device, device_type, cmd_args)
            topp.run()
            topp.print_score()
            return
        case "--config" | "-c":
            working_directory_path = ""
            if (cmd_args != None):
                working_directory_path = f"{DATA_PATH}/{cmd_args[0]}/"
            configuration = functionality.data.parse_json(working_directory_path, file_name="configuration")
            functionality.data.print_json(name="configuration", data=configuration)
            return
        case _:
            print("Error during command matching 'Command not found'")
            functionality.cli.print_commands()
            return


# branch if program is run through 'python main.py'
if __name__ == "__main__":
    main()
