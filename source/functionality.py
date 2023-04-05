# external libraries
import json
import math


def parse_arguments(args: list[str]):
    error = None
    cmd = None
    cmd_args = None
    args_size = len(args)
    if (args_size <= 0):
        error = "No arguments given"
        return (error, cmd, cmd_args)
    if (args_size > 2):
        error = "Too many arguments given, require between 1 and 2"
        return (error, cmd, cmd_args)
    cmd = args[0]
    if (args_size == 2):
        cmd_args = args[1]
    return (error, cmd, cmd_args)


def print_commands():
    msg = "\nList of commands:\n"
    msg += "\t'--help' or '-h': Shows this information\n"
    msg += "\t'--alphahex' or '-ah': Starts the alphahex program, takes parameters from 'configuration.json' file\n"
    msg += "\t'--tournament' or '-t': Starts the tournament program, requries a second argument with the directory name in 'data/'\n"
    msg += "\t'--config' or '-c': Prints current configuration\n"
    print(msg)


def parse_json(directory_path: str = "", file_name: str = "configuration") -> dict[str, any]:
    with open(directory_path + file_name + ".json", "r") as file:
        return json.load(file)


def print_json(name: str, data: dict[str, any]):
    print("\n" + name + ": " + json.dumps(data, indent=4, sort_keys=True) + "\n")


def action_from_visit_distribution(visit_distribution: list[float], board_size: int) -> tuple[int, int]:
    value = max(visit_distribution)
    index = visit_distribution.index(value)
    return index_to_action(index, board_size)


def action_to_index(action: tuple[int, int]) -> int:
        row, column = action
        return 2 * row + (column - 1)


def index_to_action(index: int, board_size: int) -> tuple[int, int]:
    row = math.floor(index / board_size)
    column = index % board_size
    return (row, column)
