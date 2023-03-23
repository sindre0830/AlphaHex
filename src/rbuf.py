class RBUF():
    def __init__(self):
        self.states: list[list[int]]
        self.actions: list[int]

    def clear(self):
        self.states.clear()
        self.actions.clear()

    def add(self, state, action):
        self.states.append(state)
        self.actions.append(action)
