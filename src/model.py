class Model():
    def __init__(
            self,
            learning_rate: float,
            hidden_layer_size: int,
            neurons_per_layer: int,
            activation_function,
            optimizer_function,
        ):
        pass

    def fit(self, player: bool, board_state: list[list[int]], y: list[int]):
        pass

    def predict(self, player: bool, board_state: list[list[int]]):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str, filename: str):
        pass
