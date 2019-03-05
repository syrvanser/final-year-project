import logging
import random
import re
import time
from games import Game
from mcts import *
import multiprocessing as mp

class RandomAgent:
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def act(self, game):
        actions = game.game_state.allowed_actions_matrix()
        action_pool = Game.action_matrix_to_array(actions)
        self.logger.debug(action_pool)
        action = random.choice(action_pool)
        game.take_action(action)
        
class HumanAgent:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def act(self, game):
        actions = game.game_state.allowed_actions_matrix()
        print(game.game_state.print())
        drop_regex = re.compile(r'[G,P,B,R,S]\s[0-4]\s[0-4]')
        move_regex = re.compile(r'[0-4]\s[0-4]\s[0-4]\s[0-4](\sD)?')
        while True:
            '''action_pool = []
            for y in range(0, game.BOARD_Y):
                for x in range(0, game.BOARD_X):
                    for z in range(0, len(actions[0][0])):
                        if(actions[y][x][z] == 1):
                            action_pool.append((x, y, z))
            print(action_pool)'''
            choice = input('> ')
            if(re.match(drop_regex, choice)):
                piece, x, y = choice.split(' ')
                x = int(x)
                y = int(y)
                z = Game.QUEEN_ACTIONS + Game.KNIGHT_ACTIONS + Game.PR_QUEEN_ACTIONS + Game.PR_KNIGHT_ACTIONS + Game.HAND_ORDER.index(piece)
                if(actions[y][x][z]==1):
                    game.take_action((x,y,z))
                    break
                else:
                    print('Illegal drop')
            elif(re.match(move_regex, choice)):
                x, y, new_x, new_y = tuple(map(int, choice.split(' ')))
                magnitude, direction = Game.get_direction(x,y,new_x,new_y)
                if(choice[-1]!='P'):
                    z = direction * (Game.MAX_MOVE_MAGNITUDE)+(magnitude-1)
                else:
                    z = Game.QUEEN_ACTIONS + Game.KNIGHT_ACTIONS + direction * (Game.MAX_MOVE_MAGNITUDE)+(magnitude-1)
                #print('trying (x, y, z): (' + str(x) + ', ' + str(y) + ', ' + str(z) + ')')
                if(actions[y][x][z]==1):
                    game.take_action((x,y,z))
                    break
                else:
                    print('Illegal move')
                    
            else:
                print('Wrong Input')

class BasicMCTSAgent:
    def __init__(self, limit, max_depth, use_timer=False):
        self.limit = limit
        self.use_timer = use_timer
        self.MCTS = BasicMCTS(nnet = 0, max_depth = max_depth)
        self.logger = logging.getLogger(self.__class__.__name__)


    def act(self, game):
        games = 0
        #if game.game_state.game_ended():
            
        if(self.use_timer == True):
            begin = time.time()
            games = 0
            while time.time() - begin < self.limit:
                self.logger.debug('Depth: #{0}'.format(games))
                self.MCTS.search(game)
                games += 1
        else: 
            for i in range(self.limit):
                self.logger.debug('Playout: #{0}'.format(i))
                self.MCTS.search(game)
        
        #action_pool = Game.action_matrix_to_array(game.game_state.allowed_actions())
        #states_pool = [Game.next_state(game.game_state, action) for action in action_pool]
        #self.logger.debug('action pool size: ' + str(len(action_pool)))
        states_pool =  game.game_state.next_states_array()


        #for i in range(0, len(action_pool)):
            #self.logger.debug(str(action_pool[i]) + ' Percentage: ' +  str(self.MCTS.q.get(states_pool[i], 0) / self.MCTS.n.get((states_pool[i]), 1)))   
        #logger.debug(action_pool)
        max_pr = 0
        next_state = states_pool[0]
        for state in states_pool:
            percentage = self.MCTS.q.get(state, 0) / self.MCTS.n.get((state), 1)
            if max_pr < percentage:
                max_pr = percentage
                next_state = state

        #percentage, next_state = max((self.MCTS.n.get(state, 0) / self.MCTS.q.get((state), 1)) for state in action_pool)
        game.move_to_next_state(next_state)
'''
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
        game.move_to_next_state(next_state)
'''

class NNetMCTSAgent:
    def __init__(self, limit, max_depth, nnet):
        self.limit = limit
        self.nnet = nnet
        self.MCTS = BasicMCTS(nnet = 0, max_depth = max_depth)
        self.logger = logging.getLogger(self.__class__.__name__)


    def act(self, game):
         
        for i in range(self.limit):
            self.logger.debug('Playout: #{0}'.format(i))
            self.MCTS.search(game)
        
        states_pool =  game.game_state.next_states_array()

        max_pr = 0
        next_state = states_pool[0]
        for state in states_pool:
            percentage = self.MCTS.q.get(state, 0) / self.MCTS.n.get((state), 1)
            if max_pr < percentage:
                max_pr = percentage
                next_state = state

        game.move_to_next_state(next_state)