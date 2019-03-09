import logging
import time

from agents.agent import Agent
from mcts.basic_mcts import BasicMCTS

logger = logging.getLogger(__name__)


class BasicMCTSAgent(Agent):
    def __init__(self, limit, max_depth, use_timer=False):
        super().__init__()
        self.limit = limit
        self.use_timer = use_timer
        self.MCTS = BasicMCTS(nnet=0, max_depth=max_depth)

    def act(self, game):
        # if game.game_state.game_ended():

        if self.use_timer:
            begin = time.time()
            games = 0
            while time.time() - begin < self.limit:
                logger.debug('Depth: #{0}'.format(games))
                self.MCTS.search(game)
                games += 1
        else:
            for i in range(self.limit):
                logger.debug('Playout: #{0}'.format(i))
                self.MCTS.search(game)

        # action_pool = Game.action_matrix_to_array(game.game_state.allowed_actions())
        # states_pool = [Game.next_state(game.game_state, action) for action in action_pool]
        # self.logger.debug('action pool size: ' + str(len(action_pool)))
        states_pool = game.game_state.next_states_array()

        # for i in range(0, len(action_pool)):
        # self.logger.debug(str(action_pool[i]) + ' Percentage: ' +  str(self.MCTS.q.get(states_pool[i], 0) / self.MCTS.n.get((states_pool[i]), 1)))
        # logger.debug(action_pool)

        next_state = max(states_pool, key=(lambda s: self.MCTS.q.get(
            s, 0) / self.MCTS.n.get(s, 1)))  # select state with max q/n ratio

        # percentage, next_state = max((self.MCTS.n.get(state, 0) / self.MCTS.q.get((state), 1)) for state in action_pool)
        game.move_to_next_state(next_state)
