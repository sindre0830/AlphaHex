from game_manager.nim import *
from functionality import *

def main():
    """
    Main program.
    """
    game_manager = Nim()
    game_manager.move(cantor_encode(0, 1), 0)
    game_manager.visualize_board()
    game_manager.move(cantor_encode(1, 2), 1)
    game_manager.visualize_board()
    game_manager.move(cantor_encode(2, 3), 0)
    game_manager.visualize_board()
    game_manager.move(cantor_encode(3, 3), 1)
    game_manager.visualize_board()
    print(game_manager.winner)
    


# run main program when file is executed
if __name__ == "__main__":
    main()
