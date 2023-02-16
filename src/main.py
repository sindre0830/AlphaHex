from game_manager.nim import *
from functionality import *

def main():
    """
    Main program.
    """
    game_manager = Nim()
    print(len(game_manager.get_legal_moves()))
    game_manager.move(cantor_encode(0, 1), 0)
    print(len(game_manager.get_legal_moves()))
    game_manager.move(cantor_encode(1, 2), 1)
    print(len(game_manager.get_legal_moves()))
    game_manager.move(cantor_encode(2, 3), 0)
    print(game_manager.get_legal_moves())
    game_manager.move(cantor_encode(3, 4), 1)
    print(len(game_manager.get_legal_moves()))
    print(cantor_decode(32))


# run main program when file is executed
if __name__ == "__main__":
    main()
