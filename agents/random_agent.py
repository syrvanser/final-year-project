import logging
import random

from agents import Agent
from games import MiniShogiGameState


class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    def act(self, game):
        actions = game.game_state.allowed_actions_matrix()
        action_pool = MiniShogiGameState.action_matrix_to_action_array(actions)
        logging.debug(action_pool)
        action = random.choice(action_pool)
        game.take_action(action)
