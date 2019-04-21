import logging
import math
from math import sqrt
from operator import add

import numpy as np

import config
from games import MiniShogiGame, MiniShogiGameState
from mcts import MCTS


EPS = 1e-8


class NNetMCTS(MCTS):

    def __init__(self, nnet):  # args = numMCTSSims
        self.nnet = nnet
        self.c_puct = config.args.c_puct
        self.n_sa = {}  # how many times a has been taken from state s
        self.q_sa = {}  # q value for taking a in state s
        self.p_s = {}  # matrix list with action probs
        self.n_s = {}  # how many times s has been visited
        # self.action_matrices = {}   # dict with action matrices
        self.action_arrays = {}
        self.max_depth = config.args.max_depth

    def get_action_probs(self, state, tau=1):
        action_pool = self.action_arrays[state]
        freq = [self.n_sa[(state, action)] if (state, action) in self.n_sa else 0 for action in action_pool]

        if tau == 0:
            max_action_prob = max(freq)
            probs = [0] * len(freq)
            probs[freq.index(max_action_prob)] = 1
            return probs

        freq = [x ** (1. / tau) for x in freq]

        probs = [n / float(sum(freq)) for n in freq]

        return probs

    def search(self, game):

        last_state = game.game_state
        action_history = []
        v = 0

        for i in range(self.max_depth):
            # logging.debug('\t\t\tMCTS Depth level: #{0}'.format(i))
            current_state = last_state
            # logging.debug(current_state.print_state(i))

            if current_state.game_ended():
                # logging.debug('Found terminal node!')
                v = -1
                break

            if current_state not in self.p_s:
                # leaf node
                self.p_s[current_state], v = self.nnet.predict(current_state)
                self.p_s[current_state] = self.p_s[current_state].reshape(
                    MiniShogiGame.ACTION_STACK_HEIGHT, MiniShogiGame.BOARD_Y, MiniShogiGame.BOARD_X)
                valid_moves = current_state.allowed_actions_matrix()
                # masking invalid moves element wise multiplication
                self.p_s[current_state] = self.p_s[current_state] * valid_moves
                sum_prob_vector = np.sum(self.p_s[current_state])
                if sum_prob_vector > 0:

                    self.p_s[current_state] /= sum_prob_vector  # renormalize
                    #logging.info(self.p_s[current_state])
                else:
                    # if all valid moves were masked make all valid moves equally probable
                    logging.warning(str(id(self)) + ': all valid moves were masked, making all equally probable')
                    self.p_s[current_state] = valid_moves
                    # renormilise
                    self.p_s[current_state] = self.p_s[current_state] / np.sum(self.p_s[current_state])

                self.action_arrays[current_state] = MiniShogiGameState.action_matrix_to_action_array(valid_moves)
                self.n_s[current_state] = 0
                break

            action_pool_array = self.action_arrays[current_state]
            p_s = self.p_s[current_state]
            if i == 0: #if root node
                noise = np.random.dirichlet([config.args.dir_alpha] * len(action_pool_array))
                for index, a in enumerate(action_pool_array):
                    p_s[a] = (1-config.args.dir_epsilon) * p_s[a] + config.args.dir_epsilon * noise[index]

            next_action = max(action_pool_array, key=(
                lambda a: (self.q_sa[(current_state, a)] + self.c_puct * p_s[a] *
                           sqrt(self.n_s[current_state]) / (1 + self.n_sa[(current_state, a)])
                           if (current_state, a) in self.q_sa
                           else self.c_puct * p_s[a] * math.sqrt(self.n_s[current_state] + EPS))))

            '''logging.debug('Selecting action with q={0} n={1} p={2}'.format(
                self.q_sa.get((current_state, next_action), 'n/a'), self.n_sa.get((current_state, next_action), 'n/a'),
                self.p_s[current_state][next_action]))'''

            next_state = MiniShogiGameState.action_to_state(current_state, next_action)
            last_state = next_state
            action_history.append((current_state, next_action))

        for (parent, action) in reversed(action_history):
            if (parent, action) not in self.q_sa:
                self.q_sa[(parent, action)] = -v  # v is value of next state, hence using -v for this
                v = -v
                self.n_sa[(parent, action)] = 1
            else:
                self.q_sa[(parent, action)] = (self.n_sa[(parent, action)] *
                                               self.q_sa[(parent, action)] + (-v)) / (self.n_sa[(parent, action)] + 1)
                v = -v
                self.n_sa[(parent, action)] += 1
            self.n_s[parent] += 1

