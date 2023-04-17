# device type
GPU_DEVICE = 'cuda:0'
CPU_DEVICE = 'cpu'
# parameters
VERBOSE_TRAINING = False
BATCH_SIZE = 32
TOURNAMENT_GAMES = 25
TOURNAMENT_VISUALIZATION = False
# paths
DATA_PATH = "data"
# directions
NEIGHBOUR_DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]
BRIDGE_DIRECTIONS = [(-1, -1), (1, 1), (1, -2), (2, -1), (-2, 1), (-1, 2)]
# cell type
PLAYER_1 = 1
PLAYER_2 = 2
EMPTY = 0
TEST = -1
# conditions
TIE = 0
DNF = -1
