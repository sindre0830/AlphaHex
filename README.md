# AlphaHex
AlphaHex is an implementation of the architecture used in AlphaGo, but adapted for the game Hex. This repository offers a deep reinforcement learning approach to playing Hex by combining deep neural networks with Monte Carlo Tree Search (MCTS).

### Run program
0. Install required python version **3.10**
1. Install required packages `pip install -r source/requirements.txt` (We recommend using virtual environment, follow guide under **Virtual Environment Setup** below and skip this step)
3. Change directory `cd source`
4. Run program `py main.py --help`

### Virtual Environment Setup
#### Windows
0. Get the package `pip install virtualenv`
1. Create a new empty instance of python 3.10 environment `py -3.10 -m venv ./.venv`
2. Activate the environment `source .venv/Scripts/activate`
3. Install the packages required by this project `pip install -r source/requirements.txt`

### Project
This was a project for the IT3105 course at NTNU.

#### Tasks
- Add main loop
- Add Hex game manager
- Add hidden layers based parameter
- Add activation function based on parameter
- Add optimizer function based on parameter
- Move parameters to JSON file and load them on startup
- Add model save interval
- Add RBUF
- Add tournament program between the different models
    - The aim is to see a gradual improvement from the first save to the last save
- Add game state visualization
    - Used to verify the final game state

#### Variables
- I_s = Save interval
- B_a = Actual game board
- B_mc = Monte Carlo game board
- S_init = Starting board state
- S_s = Successor state
- G_a = Game (actual)
- G_s = Search game
- P_t = Tree policy
- L = Leaf node
- F = Final state node
- D = Distribution of visit counts along all arcs emanating from root
- RBUF = Replay buffer used to train the NN
- M_a = Actual move
- Default policy = ANET
- Tree policy = Upper confidence bound

#### Pseudocode
```
1. I_s = save interval for ANET (the actor network) parameters
2. Clear Replay Buffer (RBUF)
3. Randomly initialize parameters (weights and biases) of ANET
4. For G_a in number actual games:
    - Initialize the actual game board (B_a) to an empty board.
    - S_init ← starting board state
    - Initialize the Monte Carlo Tree (MCT) to a single root, which represents S_init
    - While B_a not in a final state:
        - Initialize Monte Carlo game board (B_mc) to same state as root.
        - For G_s in number search games:
            - Use tree policy P_t to search from root to a leaf (L) of MCT. Update B_mc with each move.
            - Use ANET to choose rollout actions from L to a final state (F). Update B_mc with each move.
            - Perform MCTS backpropagation from F to root.
        - next G_s
        - D = distribution of visit counts in MCT along all arcs emanating from root.
        - Add case (root, D) to RBUF
        - Choose actual move (M_a) based on D
        - Perform M_a on root to produce successor state S_s
        - Update B_a to S_s
        - In MCT, retain subtree rooted at S_s; discard everything else.
        - root ← S_s
    - Train ANET on a random minibatch of cases from RBUF
    - if G_a modulo I_s == 0:
        - Save ANET's current parameters for later use in tournament play.
5. next G_a
```
