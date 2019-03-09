from abc import ABC, abstractmethod


class Game(ABC):

    def __init__(self):
        super(Game, self).__init__()

    @abstractmethod
    def take_action(self, action):
        pass

    @abstractmethod
    def move_to_next_state(self, next_state):
        pass
