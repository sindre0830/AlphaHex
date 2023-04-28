from state_manager import StateManager
from anet import ANET
import random

def random_match(anet: ANET, grid_size: int):
    wins_player1 = 0
    wins_player2 = 0
    total_games = 25
    # as player 1
    for _ in range(total_games):
        local_state = StateManager()
        local_state.initialize_state(grid_size)
        anet_player = 1
        while not local_state.terminal():
            if local_state.player == anet_player:
                state = (local_state.grid, local_state.player)
                probability_distribution = anet.predict(state, local_state.illegal_actions())
                local_state.apply_action_from_distribution(probability_distribution, deterministic=True)
            else:
                action = random.choice(local_state.legal_actions())
                local_state.apply_action(action)
        if local_state.determine_winner() == anet_player:
            wins_player1 += 1
    # as player 2
    for _ in range(total_games):
        local_state = StateManager()
        local_state.initialize_state(grid_size)
        anet_player = 2
        while not local_state.terminal():
            if local_state.player == anet_player:
                state = (local_state.grid, local_state.player)
                probability_distribution = anet.predict(state, local_state.illegal_actions())
                local_state.apply_action_from_distribution(probability_distribution, deterministic=True)
            else:
                action = random.choice(local_state.legal_actions())
                local_state.apply_action(action)
        if local_state.determine_winner() == anet_player:
            wins_player2 += 1
    winrate = (wins_player1 + wins_player2) / (total_games * 2)
    print(f"\tWinrate {(winrate):0.2f}")
    return winrate
