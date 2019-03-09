"""import logging
from math import log, sqrt
from random import choice

from mcts.mcts import MCTS

EPS = 1e-8

logger = logging.getLogger(__name__)


class ParallelMCTS(MCTS):

    def __init__(self, nnet, max_depth):  # args = numMCTSSims
        self.nnet = nnet
        self.c_puct = 1.4
        self.max_depth = max_depth

    def search(self, game):
        visited_states = set()
        states_history_copy = game.state_history[:]
        expand = True
        for i in range(self.max_depth):
            logger.debug('Depth level: #{0}'.format(i))
            current_state = states_history_copy[-1]
            logger.debug(current_state.print(i))

            # actions = current_state.allowed_actions()
            # action_pool = Game.action_matrix_to_array(actions)
            # states_pool = [Game.next_state(current_state, action) for action in action_pool]

            states_pool = current_state.next_states_array()
            # print('{0} {1}'.format(len(states_pool), len(states_pool2)))
            explored = sum(el in n for el in states_pool)
            logger.info(
                'Actions explored: {0}/{1}'.format(explored, len(states_pool)))

            if all(n.get(s) for s in states_pool):
                log_total = log(
                    sum(n[s] for s in states_pool))

                next_state = max(states_pool, key=(lambda s: (
                                                                     q[s] / n[s]) + self.c_puct * sqrt(
                    log_total / n[s])))
                logger.info('Selecting action with q={0} n={1}'.format(
                    q[next_state], n[next_state]))
                # print (value)
            else:
                logger.info('Not enough data, choosing at random!')
                next_state = choice(states_pool)

            # action = choice(action_pool)
            # next_state = Game.next_state(current_state, action)
            # next_state = choice(states_pool)
            states_history_copy.append(next_state)

            if expand and next_state not in n:
                expand = False
                n[next_state] = 0
                q[next_state] = 0

            visited_states.add(next_state)

            if next_state.game_ended():
                logger.debug('Found terminal node!')
                break

        for state in visited_states:
            if state not in n:
                continue

            n[state] += 1
            # print(self.n[state])

            if state.game_ended() and state.colour != game.game_state.colour:
                q[state] += 1
        return (n, q)
"""