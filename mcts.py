import math
import numpy as np
from random import choice
from games import Game, GameState
from math import log, sqrt
import logging
logger = logging.getLogger(__name__)

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, nnet, max_depth): #args = numMCTSSims
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
            logger.debug('Depth level: #{0}'.format(i))
            current_state = states_history_copy[-1]
            logger.debug(current_state.print())

            #actions = current_state.allowed_actions()
            #action_pool = Game.action_matrix_to_array(actions)
            #states_pool = [Game.next_state(current_state, action) for action in action_pool]

            states_pool =  Game.next_states_array(current_state)
           # print('{0} {1}'.format(len(states_pool), len(states_pool2)))
            
            if all(self.n.get(s) for s in states_pool):
                log_total = log(
                    sum(self.n[s] for s in states_pool))
                
                max_value = -1
                next_state = states_pool[0]
                for s in states_pool:
                    value = (self.q[s] / self.n[s]) + self.c_puct * sqrt(log_total / self.n[s])
                    if value > max_value:
                        max_value = value
                    next_state = s

                #print (value)
                #print(next_state)
            else:
                next_state = choice(states_pool)
            

            #action = choice(action_pool)
            #next_state = Game.next_state(current_state, action)
            #next_state = choice(states_pool)
            states_history_copy.append(next_state)

            if expand and next_state not in self.n:
                expand = False
                self.n[next_state] = 0
                self.q[next_state] = 0

            visited_states.add(next_state)

            if next_state.game_ended():
                logger.debug('Found terminal node!')
                break
        
        for state in visited_states:
            if state not in self.n:
                continue

            self.n[state] += 1
            #print(self.n[state])
            
            if state.game_ended() and state.colour != game.game_state.colour:
                self.q[state] +=1

