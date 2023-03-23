# AlphaHex
### Virtual Environment Setup
#### Windows
0. Get the package `pip install virtualenv`
1. Create a new empty instance of python 3.10 environment `py -3.10 -m venv ./.venv`
2. Activate the environment `source .venv/Scripts/activate`
3. Install the packages required by this project `pip install -r requirements.txt`

### Project
#### Tasks
- Add main loop (T)
- Add Hex game manager (T)
- Add model save interval (S)
- Add RBUF (S)
- Add model visualization (S)
- Add tournament program between the different saves
    - The aim is to see a gradual improvement from the first save to the last save
- Add the thing where different AIs play against themselves
- Add the whole tournament thing

- Add game state visualization (S)
    - Used to verify the final game state
- Find out what we actually need for the project (S, T)
- Move parameters to JSON file and load them on startup (T)

#### Variables
I_s = Save interval
B_a = Actual game board
B_mc = Monte Carlo game board
S_init = Starting board state
S_s = Successor state
G_a = Game (actual)
G_s = Search game
P_t = Tree policy
L = Leaf node
F = Final state node
D = Distribution of visit counts along all arcs emanating from root
RBUF = Replay buffer used to train the NN
M_a = Actual move
Default policy = ANET
Tree policy = Upper confidence bound

#### Pseudocode
```
1. I_s = save interval for ANET (the actor network) parameters
2. Clear Replay Buffer (RBUF)
3. Randomly initialize parameters (weights and biases) of ANET
4. For g_a in number actual games:
    - Initialize the actual game board (B_a) to an empty board.
    - S_init ← starting board state
    - Initialize the Monte Carlo Tree (MCT) to a single root, which represents sinit
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

#### Explanation (not entirely correct)
1. Initialization: At the start of the game, the AI agent initializes a Monte Carlo (MC) tree with the root node representing the current game state. The root node has a visit count of 0 and a value of 0.

2. MCTS simulation: The AI agent starts the Monte Carlo Tree Search (MCTS) algorithm to simulate potential moves. The MCTS algorithm consists of four steps:

a. Selection: The AI agent selects a node in the MC tree using the Upper Confidence Bound (UCB) algorithm. The UCB algorithm balances exploration and exploitation by choosing nodes with high values and high uncertainty (i.e., high visit count). The AI agent continues selecting nodes until it reaches a leaf node in the MC tree.

b. Expansion: The AI agent expands the selected leaf node by adding child nodes representing each possible action from that node. Each child node is initialized with a visit count of 0 and a value of 0.

c. Simulation: The AI agent simulates a random game (i.e., rollout) from each child node until it reaches a final game state. The rollout policy is determined by the ANET, which makes the AI agent choose actions that maximize the chance of winning the game. The AI agent then calculates a reward signal based on the outcome of the game (win, loss, or draw).

d. Backpropagation: The AI agent backpropagates the reward signal from the final game state back to the root node, updating the visit count and value of each node along the path taken during the simulation.

3. Actual move: After performing a certain number of MCTS simulations, the AI agent chooses an actual move to make based on the MC tree. The AI agent chooses the child node with the highest visit count as the next move. This move is then played in the actual game.

4. Updating target policy: After playing the actual move, the AI agent updates the target policy by using supervised learning. The training cases are pairs (s,D), where s is a game state and D is a probability distribution over the possible actions in that state. The probability distribution is derived from the normalized visit counts of the child nodes of the root node in the MC tree. The AI agent stores these training cases in a Replay Buffer.

5. Training ANET: At the end of each game (episode), the AI agent randomly selects a minibatch of training cases from the Replay Buffer and trains the ANET using supervised learning. The ANET learns to predict the probability distribution D from a given game state.

6. x|Repeat: The AI agent repeats steps 2-5 for each subsequent move until the game is over. At the end of the game, the AI agent updates the MC tree with the outcome of the game and resets it to the initial state for the next game.