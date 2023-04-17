def parse_arguments(args: list[str]):
    error = None
    cmd = None
    cmd_args = None
    args_size = len(args)
    if (args_size <= 0):
        error = "No arguments given"
        return (error, cmd, cmd_args)
    cmd = args[0]
    if args_size >= 2:
        cmd_args = args[1:]
    return (error, cmd, cmd_args)


def print_commands():
    msg = "\nList of commands:\n"
    msg += "\t'--help' or '-h': Shows this information\n"
    msg += "\t'--alphahex' or '-ah': Starts the alphahex program, takes parameters from 'configuration.json' file\n"
    msg += "\t'--tournament' or '-t': Starts the tournament program, requries a second argument with the directory name in 'data/'\n"
    msg += "\t'--config' or '-c': Prints current configuration\n"
    print(msg)
