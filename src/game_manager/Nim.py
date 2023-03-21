class Nim:
    def __init__(self, initial_piles: list[int]):
        self.initial_piles = initial_piles

    @property
    def initial_state(self) -> list[int]:
        return self.initial_piles.copy()

    def is_terminal(self, state: list[int]) -> bool:
        return sum(state) == 0

    def get_legal_actions(self, state: list[int]) -> list[tuple[int, int]]:
        legal_actions = []
        for pile_idx, pile_size in enumerate(state):
            for count in range(1, pile_size + 1):
                legal_actions.append((pile_idx, count))
        return legal_actions

    def next_state(self, state: list[int], action: tuple[int, int]) -> list[int]:
        pile_idx, count = action
        next_state = state.copy()
        next_state[pile_idx] -= count
        return next_state

    def is_valid_action(self, state: list[int], action: tuple[int, int]) -> bool:
        pile_idx, count = action
        return 0 <= pile_idx < len(state) and 0 < count <= state[pile_idx]