# internal libraries
from constants import (
    GPU_DEVICE,
    CPU_DEVICE
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
    args = sys.argv[1:]
    print(args)


# branch if program is run through 'python main.py'
if __name__ == "__main__":
    main()
