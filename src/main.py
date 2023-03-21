from game_manager.nim import *
from functionality import *
from mcts import *
from anet import *

def main():
    """
    Main program.
    """
    game_manager = Nim()
    anet = ANET()
    mcts = MCTS(game_manager, 500, 3, anet)
    mcts.root.increment_visits()
    mcts.node_expansion(mcts.root)
    number = mcts.evaluate_node(mcts.root.children[0])
    print(number)


# run main program when file is executed
if __name__ == "__main__":
    main()
