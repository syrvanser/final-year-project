import logging
import random

from games.mini_shogi_game import MiniShogiGame

logger = logging.getLogger(__name__)


class RandomAgent:

    def act(self, game):
        actions = game.game_state.allowed_actions_matrix()
        action_pool = MiniShogiGame.action_matrix_to_array(actions)
        logger.debug(action_pool)
        action = random.choice(action_pool)
        game.take_action(action)
