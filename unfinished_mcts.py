'''

import math
import numpy as np
from games import Game, GameState
EPS = 1e-8


class State_Wrapper():
    def __init__(self, state):
        self.state = state
        self.expected_reward_Q = 0
        self.times_visited_N = 0
        self.initial_estimate_P = 0
        self.is_terminal_node = state.game_ended()
        self.moves = state.allowed_actions()
    
    def __eq__(self, other): 
        return self.state == other.state

    def __ne__(self, other): 
        return not (self == other)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, number_of_simulations, cpuct): #args = numMCTSSims
        self.game = game
        self.visited_states = []
        self.nnet = nnet
        self.number_of_simulations = number_of_simulations
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)
        self.cpuct = cpuct
        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def get_action_probability(self, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for _ in range(self.number_of_simulations):
            self.search(self.game.INITIAL_LAYOUT)

        actions = self.game.game_state.allowed_actions()

        if temp==0:
            bestA = np.argmax(actions)
            probs = [0]*len(actions)
            probs[bestA]=1
            return probs

        actions = [x**(1./temp) for x in actions]
        probs = [x/float(sum(actions)) for x in actions]
        return probs


    def search(self, state):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = State_Wrapper(state)

        if s.is_terminal_node:
            # terminal node 
            if (self.start_colour==s.state.colour):
                return -Game.game_reward() #if you're playing white and current state colour is white then you've lost
            else:
                return Game.game_reward()

        if s not in self.visited_states:
            # leaf node
            s.initial_estimate_P, v = self.nnet.predict(s)
            #valids = self.game.getValidMoves(canonicalBoard, 1)
            
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v

    @staticmethod
    def search2(game, nnet):
        state = game.game_state
        visited = []

        if game.game_state.game_ended(): return -game.game_reward()

        if state not in visited:
            visited.append(state)
            P[state], v = nnet.predict(state)
            return -v

        max_u, best_a = -float("inf"), -1
        for a in range(game.getValidActions(state)):
            u = Q[state][a] + c_puct*P[state][a]*sqrt(sum(N[state]))/(1+N[state][a])
            if u>max_u:
                max_u = u
                best_a = a
        a = best_a
        
        sp = game.nextState(state, a)
        v = search(sp, game, nnet)

        Q[state][a] = (N[state][a]*Q[state][a] + v)/(N[state][a]+1)
        N[state][a] += 1
        return -v
'''