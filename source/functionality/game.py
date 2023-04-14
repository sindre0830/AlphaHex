# internal libraries
from constants import (
    DATA_PATH
)
# external libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon
from matplotlib.collections import LineCollection


def action_from_visit_distribution(visit_distribution: list[float], board_size: int) -> tuple[int, int]:
    value = max(visit_distribution)
    index = visit_distribution.index(value)
    return index_to_action(index, board_size)


def action_to_index(action: tuple[int, int], width: int) -> int:
    row, column = action
    return (row * width) + column


def index_to_action(index: int, width: int) -> tuple[int, int]:
    row = math.floor(index / width)
    column = index - (row * width)
    return (row, column)


def opposite_player(current_player: int) -> int:
    return 2 if current_player == 1 else 1


def animate_game(save_directory_name: str, board_history: list[list[list[int]]], iteration: int):
    def animate(i):
        # clear previous canvas
        ax.cla()
        # draw polygons
        for row in range(board_size):
            for col in range(board_size):
                x, y = col + row, -col + row
                patch = RegularPolygon((x / 1.1, y / 1.9), numVertices=6, radius=0.6, orientation=np.pi / 2, facecolor=player_colors[board_history[i][row][col]], edgecolor="black", linewidth=1)
                ax.add_patch(patch)
        # draw lines
        offset = 0.8
        edge_states = [
            ((0 - offset, 0 - offset), (0 - offset, board_size - 1 + offset), 1),
            ((0 - offset, board_size - 1 + offset), (board_size - 1 + offset, board_size - 1 + offset), 2),
            ((board_size - 1 + offset, board_size - 1 + offset), (board_size - 1 + offset, 0 - offset), 1),
            ((board_size - 1 + offset, 0 - offset), (0 - offset, 0 - offset), 2)
        ]
        lines = []
        line_colors = []
        for edge_state in edge_states:
            (row_1, col_1), (row_2, col_2), player = edge_state
            x_1, y_1 = col_1 + row_1, -col_1 + row_1
            x_2, y_2 = col_2 + row_2, -col_2 + row_2
            lines.append([(x_1 / 1.1, y_1 / 1.9), (x_2 / 1.1, y_2 / 1.9)])
            line_colors.append(player_colors[player])
        ax.add_collection(LineCollection(lines, colors=line_colors, linewidths=4))
        # set size and apspect ratio of canvas
        ax.set_box_aspect(1)
        size = -0.5
        ax.set_xlim(-1 + size, 2 * board_size - 1 - size)
        ax.set_ylim(-board_size + size, board_size - size)
        ax.axis("off")
        return [ax]
    # duplicate last state
    last_element = board_history[-1]
    duplicates = [last_element] * 4
    board_history = board_history + duplicates
    player_colors = {-1: "green", 0: "none", 1: "red", 2: "blue"}
    # create gif
    fig, ax = plt.subplots(figsize=(10, 10))
    board_size = len(board_history[0])
    ani = animation.FuncAnimation(fig, animate, frames=len(board_history), interval=500)
    # save gif
    ani.save(f"{DATA_PATH}/{save_directory_name}/visualization_{iteration}.gif", writer="pillow")
