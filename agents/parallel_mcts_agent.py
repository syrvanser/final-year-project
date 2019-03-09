"""import logging
import random
import re
import time
from copy import deepcopy
from games import Game
from collections import deque
from mcts import *
import numpy as np
import multiprocessing as mp
import time
import os
import sys
from pickle import Pickler, Unpickler


class ParallelMCTSAgent:
    def __init__(self, limit, max_depth, processes=8):
        self.limit = limit
        self.processes = processes
        self.MCTS = ParallelMCTS(nnet = 0, max_depth = max_depth)
        self.logger = logging.getLogger(self.__class__.__name__)

    def act(self, game):

        #if game.game_state.game_ended():

        with mp.Pool(processes=self.processes) as pool:
            for i in range(0, self.limit):
                self.logger.debug('Playout: #{0}'.format(i))
                temp = [pool.apply_async(self.MCTS.search, (game,n,q)) for i in range(self.limit)] #FIX LATER
                results = [t.get() for t in temp]
                #pool.apply_async(self.MCTS.search, (game,))

        #action_pool = Game.action_matrix_to_array(game.game_state.allowed_actions())
        #states_pool = [Game.next_state(game.game_state, action) for action in action_pool]
        #self.logger.debug('action pool size: ' + str(len(action_pool)))
        states_pool =  game.game_state.next_states_array()


        #for i in range(0, len(action_pool)):
            #self.logger.debug(str(action_pool[i]) + ' Percentage: ' +  str(self.MCTS.q.get(states_pool[i], 0) / self.MCTS.n.get((states_pool[i]), 1)))
        #logger.debug(action_pool)

        next_state = max(states_pool, key = (lambda s: q.get(s, 0) / n.get((s), 1))) #select state with max q/n ratio

        #percentage, next_state = max((self.MCTS.n.get(state, 0) / self.MCTS.q.get((state), 1)) for state in action_pool)
        game.move_to_next_state(next_state)"""
