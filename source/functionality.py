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
