from random import choice

from math import log, sqrt

from mcts import MCTS


class BasicMCTS(MCTS):

    def __init__(self, nnet, max_depth):  # args = numMCTSSims
        self.nnet = nnet
        self.n = {}
        self.q = {}
        self.c_puct = 1.4
        self.max_depth = max_depth

    def search(self, game):
        visited_states = set()
        states_history_copy = game.state_history[:]
        expand = True
        for i in range(self.max_depth):
            current_state = states_history_copy[-1]
            states_pool = current_state.next_states_array()
            if all(self.n.get(s) for s in states_pool):
                log_total = log(
                    sum(self.n[s] for s in states_pool))

                next_state = max(states_pool, key=(lambda s: (self.q[s] / self.n[s]) + self.c_puct * sqrt(
                    log_total / self.n[s])))
            else:
                next_state = choice(states_pool)

            states_history_copy.append(next_state)

            if expand and next_state not in self.n:
                expand = False
                self.n[next_state] = 0
                self.q[next_state] = 0

            visited_states.add(next_state)

            if next_state.game_ended():
                break

        for state in visited_states:
            if state not in self.n:
                continue

            self.n[state] += 1
            if state.game_ended() and state.colour != game.game_state.colour:
                self.q[state] += 1
