class RBUF():
    def __init__(self):
        self.states: list[list[int]] = []
        self.visit_distributions: list[list[int]] = []

    def clear(self):
        self.states.clear()
        self.visit_distributions.clear()

    def add(self, state, visit_distribution):
        self.states.append(state)
        self.visit_distributions.append(visit_distribution)
