from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def act(self, game):
        pass
