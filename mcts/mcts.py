from abc import ABC, abstractmethod


class MCTS(ABC):

    @abstractmethod
    def search(self, game):
        pass
