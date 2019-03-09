import logging
import math
from math import sqrt
from random import shuffle

import numpy as np

from games.mini_shogi_game import MiniShogiGame
from mcts.mcts import MCTS

logger = logging.getLogger(__name__)
EPS = 1e-8

class NNetMCTS(MCTS):
    
    def __init__(self, nnet, args):  # args = numMCTSSims
        self.nnet = nnet
        self.c_puct = args.c_puct
        self.n = {}
        self.q = {}
        self.p = {}
        self.e = {}  # stores game.getGameEnded ended for board s
        self.valid_moves = {}
        self.max_depth = args.max_depth

    def pi(self, state_pool):
        return [self.p[state] for state in state_pool]

    def search(self, game):

        states_history_copy = game.state_history[:]
        visited_states = []
        v = 0

        for i in range(self.max_depth):
            logger.info('Depth level: #{0}'.format(i))
            current_state = states_history_copy[-1]
            logger.info(current_state.print(i))

            if current_state.game_ended():
                logger.info('Found terminal node!')
                v = -1
                break

            if current_state not in self.p:
                # leaf node
                self.p[current_state], v = self.nnet.predict(current_state)
                self.p[current_state] = self.p[current_state].reshape(
                    MiniShogiGame.ACTION_STACK_HEIGHT, MiniShogiGame.BOARD_Y, MiniShogiGame.BOARD_X)
                valid_moves = current_state.allowed_actions_matrix()
                # masking invalid moves elementwise multiplication
                self.p[current_state] = self.p[current_state] * valid_moves
                sum_prob_vector = np.sum(self.p[current_state])
                if sum_prob_vector > 0:
                    self.p[current_state] /= sum_prob_vector  # renormalize

                else:
                    # if all valid moves were masked make all valid moves equally probable
                    logger.warning(
                        'All valid moves were masked, making all equally probable')
                    self.p[current_state] = self.p[current_state] + valid_moves
                    # renormilise
                    self.p[current_state] /= np.sum(self.p[current_state])

                # self.valid_moves[current_state] = valid_moves
                self.n[current_state] = 0
                break

            states_pool = current_state.matrix_to_array(self.p[current_state])
            # self.valid_moves[current_state])

            # print (list(x[1] for x in states_pool))

            explored = sum(el[0] in self.n for el in states_pool)
            logger.info(
                'Actions explored: {0}/{1}'.format(explored, len(states_pool)))

            # action_sum = sum(self.n.get(s[0], 0) for s in states_pool)

            shuffle(states_pool)
            # state -> prob(state) from self.p[current_state]
            # state dict from action matrix with vals 

            next_state_pair = max(states_pool, key=(
                lambda s: ((self.q[s[0]] / self.n[s[0]]) + self.c_puct * s[1] * sqrt(  # change that !!!!
                    self.n[current_state] / 1 + self.n[s[0]]) if s[1] in self.q else self.c_puct * s[1] * math.sqrt(
                    self.n[current_state] + EPS))))
            next_state, next_state_prob = next_state_pair

            logger.info('Selecting action with q={0} n={1} p={2}'.format(
                self.q.get(next_state, 'n/a'), self.n.get(next_state, 'n/a'), next_state_prob))

            states_history_copy.append(next_state)
            visited_states.append(next_state)

        for state in reversed(visited_states):
            if state not in self.q:
                self.q[state] = v
                v = -v
                self.n[state] = 1
            else:
                self.q[state] = (self.n[state] *
                                     self.q[state] + v) / (self.n[state] + 1)
                v=-v
                self.n[state] += 1
