from game_manager.nim import *
from functionality import *
from mcts import *

def main():
    """
    Main program.
    """
    game_manager = Nim()
    mcts = MCTS(game_manager, 500, 3)
    game_manager.visualize_board(mcts.root.state)
    mcts.node_expansion(mcts.root)
    for node in mcts.root.children:
        game_manager.visualize_board(node.state)
        print()


# run main program when file is executed
if __name__ == "__main__":
    main()
