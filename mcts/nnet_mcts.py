import logging
import math
from math import sqrt

import numpy as np

from games.mini_shogi_game import MiniShogiGame, MiniShogiGameState
from mcts.mcts import MCTS

logger = logging.getLogger(__name__)
EPS = 1e-8


class NNetMCTS(MCTS):

    def __init__(self, nnet, args):  # args = numMCTSSims
        self.nnet = nnet
        self.c_puct = args.c_puct
        self.n_sa = {}  # how many times a has been taken from state s
        self.q_sa = {}  # q value for taking a in state s
        self.p_s = {}  # matrix list with action probs
        self.n_s = {}  # how many times s has been visited
        #self.action_matrices = {}   # dict with action matrices
        self.action_arrays = {}
        self.max_depth = args.max_depth

    def get_action_probs(self, state):
        #action_array = self.actions[state]
        action_pool = self.action_arrays[state]
        freq = [self.n_sa[(state, action)] if (state, action) in self.n_sa else 0 for action in action_pool]
        probs = [n/float(sum(freq)) for n in freq]
        return probs

    def search(self, game):

        states_history_copy = game.state_history[:]
        action_history = []
        v = 0

        for i in range(self.max_depth):
            logger.info('\t\t\tMCTS Depth level: #{0}'.format(i))
            current_state = states_history_copy[-1]
            logger.debug(current_state.print_state(i))

            if current_state.game_ended():
                logger.debug('Found terminal node!')
                v = -1
                break

            if current_state not in self.p_s:
                # leaf node
                self.p_s[current_state], v = self.nnet.predict(current_state)
                self.p_s[current_state] = self.p_s[current_state].reshape(
                    MiniShogiGame.ACTION_STACK_HEIGHT, MiniShogiGame.BOARD_Y, MiniShogiGame.BOARD_X)
                valid_moves = current_state.allowed_actions_matrix()
                # masking invalid moves elementwise multiplication
                self.p_s[current_state] = self.p_s[current_state] * valid_moves
                sum_prob_vector = np.sum(self.p_s[current_state])
                if sum_prob_vector > 0:
                    self.p_s[current_state] /= sum_prob_vector  # renormalize

                else:
                    # if all valid moves were masked make all valid moves equally probable
                    logger.warning(
                        'All valid moves were masked, making all equally probable')
                    self.p_s[current_state] = valid_moves
                    # renormilise
                    self.p_s[current_state] /= np.sum(self.p_s[current_state])

                #self.action_matrices[current_state] = valid_moves
                self.action_arrays[current_state] = MiniShogiGameState.action_matrix_to_action_array(game.game_state, valid_moves)
                self.n_s[current_state] = 0
                break

            action_pool_array = self.action_arrays[current_state]

            next_action = max(action_pool_array, key=(
                lambda a: (self.q_sa[(current_state, a)] + self.c_puct * self.p_s[current_state][a] *
                           sqrt(self.n_s[current_state]) / (1 + self.n_sa[(current_state, a)])
                           if (current_state, a) in self.q_sa
                           else self.c_puct * self.p_s[current_state][a] * math.sqrt(self.n_s[current_state] + EPS))))

            logger.debug('Selecting action with q={0} n={1} p={2}'.format(
                self.q_sa.get((current_state, next_action), 'n/a'), self.n_sa.get((current_state, next_action), 'n/a'),
                self.p_s[current_state][next_action]))

            next_state = MiniShogiGameState.action_to_state(current_state, next_action)
            states_history_copy.append(next_state)
            action_history.append((current_state, next_action))

        for (parent, action) in reversed(action_history):
            if (parent, action) not in self.q_sa:
                self.q_sa[(parent, action)] = v
                v = -v
                self.n_sa[(parent, action)] = 1
            else:
                self.q_sa[(parent, action)] = (self.n_sa[(parent, action)] *
                                               self.q_sa[(parent, action)] + v) / (self.n_sa[(parent, action)] + 1)
                v = -v
                self.n_sa[(parent, action)] += 1
            self.n_s[parent] += 1
