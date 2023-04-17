# internal libraries
from constants import (
    DATA_PATH,
    NEIGHBOUR_DIRECTIONS,
    PLAYER_1,
    PLAYER_2,
    EMPTY,
    TEST,
    TIE,
    DNF
)
import functionality.data
# external libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon
from matplotlib.collections import LineCollection


class StateManager:
    def __init__(self):
        self.grid_size: int = None
        self.grid: np.ndarray = None
        self.player: int = None
        self.action: tuple[int, int] = None
        self.grid_history: list[np.ndarray] = None
    
    def initialize_state(self, grid_size: int):
        self.grid_size = grid_size
        self.grid = np.zeros(shape=(self.grid_size, self.grid_size), dtype=np.int8)
        self.player = 1
        self.grid_history = []
        self.save_grid_to_history()
    
    def copy_state(self, state):
        self.grid_size = state.grid_size
        self.grid = state.grid.copy()
        self.player = state.player
        self.action = state.action
        self.grid_history = state.grid_history.copy()
    
    def apply_action_from_distribution(self, distribution: np.ndarray, deterministic: bool, greedy_epsilon=None):
        if deterministic:
            action = functionality.data.index_to_action(np.argmax(distribution), self.grid_size)
        else:
            if greedy_epsilon is not None:
                if random.random() < greedy_epsilon:
                    action = random.choices(population=self.legal_actions(), k=1)[0]
                else:
                    action = functionality.data.index_to_action(np.argmax(distribution), self.grid_size)
            else:
                action = random.choices(population=self.all_actions(), weights=distribution, k=1)[0]
        self.apply_action(action)

    def apply_action(self, action: tuple[int, int]):
        row, col = action
        self.grid[row][col] = self.player
        self.player = self.opponent()
        self.action = action
        self.save_grid_to_history()

    def save_grid_to_history(self):
        self.grid_history.append(self.grid.copy())
    
    def round(self) -> int:
        return len(self.grid_history)
    
    def opponent(self) -> int:
        return PLAYER_2 if self.player == PLAYER_1 else PLAYER_1
    
    def total_possible_moves(self, grid_size: int = None) -> int:
        if grid_size is None:
            grid_size = self.grid_size
        return grid_size * grid_size
    
    def all_actions(self) -> list[tuple[int, int]]:
        return [(row, col) for row in range(self.grid_size) for col in range(self.grid_size)]
    
    def legal_actions(self) -> list[tuple[int, int]]:
        indices = np.transpose(np.where(self.grid == EMPTY))
        return [(row, col) for row, col in indices]
    
    def illegal_actions(self) -> list[tuple[int, int]]:
        indices = np.transpose(np.where(self.grid != EMPTY))
        return [(row, col) for row, col in indices]
    
    def terminal(self) -> bool:
        return self.determine_winner() != DNF
    
    def determine_winner(self) -> int:
        def depth_first_search(player: int, cell: tuple[int, int], visited: set = set()) -> bool:
            row, col = cell
            if (player == PLAYER_1 and row == self.grid_size - 1) or (player == PLAYER_2 and col == self.grid_size - 1):
                return True
            visited.add(cell)
            for row_direction, col_direction in NEIGHBOUR_DIRECTIONS:
                neighbour_row, neighbour_col = row + row_direction, col + col_direction
                neighbour_cell = (neighbour_row, neighbour_col)
                if (0 <= neighbour_row < self.grid_size and 0 <= neighbour_col < self.grid_size and self.grid[neighbour_row][neighbour_col] == player and neighbour_cell not in visited):
                    if depth_first_search(player, neighbour_cell, visited):
                        return True
            return False
        for player in [PLAYER_1, PLAYER_2]:
            for i in range(self.grid_size):
                row, col = (i, 0) if player == PLAYER_2 else (0, i)
                if self.grid[row][col] == player:
                    if depth_first_search(player, cell=(row, col)):
                        return player
        for row in self.grid:
            if EMPTY in row:
                return DNF
        return TIE

    def visualize(self, save_directory_name: str, iteration: int, filename="visualization", verbose=True):
        def generate_frame(i):
            # clear previous canvas
            ax.cla()
            # draw polygons
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    x, y = col + row, -col + row
                    patch = RegularPolygon(
                        (x / 1.1, y / 1.9),
                        numVertices=6,
                        radius=0.6,
                        orientation=np.pi / 2,
                        facecolor=player_colors[grid_history[i][row][col]],
                        edgecolor="black",
                        linewidth=1
                    )
                    ax.add_patch(patch)
            # draw lines
            offset = 0.8
            lines = []
            line_colors = []
            edge_states = [
                ((0 - offset, 0 - offset), (0 - offset, self.grid_size - 1 + offset), PLAYER_1),
                ((0 - offset, self.grid_size - 1 + offset), (self.grid_size - 1 + offset, self.grid_size - 1 + offset), PLAYER_2),
                ((self.grid_size - 1 + offset, self.grid_size - 1 + offset), (self.grid_size - 1 + offset, 0 - offset), PLAYER_1),
                ((self.grid_size - 1 + offset, 0 - offset), (0 - offset, 0 - offset), PLAYER_2)
            ]
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
            ax.set_xlim(-1 + size, 2 * self.grid_size - 1 - size)
            ax.set_ylim(-self.grid_size + size, self.grid_size - size)
            ax.axis("off")
            return [ax]
        if verbose:
            print("\tVisualizing state")
        # duplicate last state so it lasts longer in the gif
        final_grid = self.grid_history[-1]
        duplicates = [final_grid] * 4
        grid_history = self.grid_history + duplicates
        # create gif
        player_colors = {TEST: "green", EMPTY: "none", PLAYER_1: "red", PLAYER_2: "blue"}
        fig, ax = plt.subplots(figsize=(10, 10))
        ani = animation.FuncAnimation(fig, generate_frame, frames=len(grid_history), interval=500)
        # save gif
        ani.save(f"{DATA_PATH}/{save_directory_name}/{filename}-{iteration}.gif", writer="pillow")
