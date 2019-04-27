from abc import ABC, abstractmethod


class GameState(ABC):

    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def from_plane_stack(cls, stack):
        pass

    @classmethod
    @abstractmethod
    def from_board(cls, args):
        pass

    @abstractmethod
    def allowed_actions_matrix(self):
        pass

    @abstractmethod
    def next_states_array(self):
        pass

    @staticmethod
    @abstractmethod
    def action_matrix_to_action_array(matrix):
        pass

    @staticmethod
    @abstractmethod
    def action_matrix_to_state_array(state, matrix):
        pass

    @staticmethod
    @abstractmethod
    def prob_list_to_matrix(pi_list, actions):
        pass

    @staticmethod
    @abstractmethod
    def action_to_state(state, action):
        pass

    @abstractmethod
    def game_ended(self):
        pass

    @staticmethod
    @abstractmethod
    def state_to_plane_stack(state):
        pass

    @staticmethod
    @abstractmethod
    def plane_stack_to_board(stack):
        pass

    @abstractmethod
    def print_state(self, level=0, flip=False):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def compare_boards(self, other):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __ne__(self, other):
        pass
