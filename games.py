import numpy as np
from collections import Counter
import copy
from hashlib import sha1
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)

class Game:
    BOARD_X = 5
    BOARD_Y = 5
    ALLOWED_REPEATS = 3
    INITIAL_LAYOUT = [['r', 'b', 's', 'g', 'k'],
                      ['.', '.', '.', '.', 'p'],
                      ['.', '.', '.', '.', '.'],
                      ['P', '.', '.', '.', '.'],
                      ['K', 'G', 'S', 'B', 'R']]

    PROMOTION_ZONE_SIZE = 1

    ORDER = ['P', 'S', 'G', 'B', 'R', '+P', '+S', '+B', '+R',
             'K', 'p', 's', 'g', 'b', 'r', '+p', '+s', '+b', '+r', 'k']
    HAND_ORDER = ['P', 'S', 'G', 'B', 'R']

    MAX_MOVE_MAGNITUDE = (max(BOARD_X, BOARD_Y))
    QUEEN_ACTIONS = 8 * MAX_MOVE_MAGNITUDE
    KNIGHT_ACTIONS = 0  # no knights in minishogi
    PR_QUEEN_ACTIONS = QUEEN_ACTIONS
    PR_KNIGHT_ACTIONS = KNIGHT_ACTIONS
    DROP = len(HAND_ORDER)
    ACTION_STACK_HEIGHT =  QUEEN_ACTIONS + KNIGHT_ACTIONS + PR_QUEEN_ACTIONS + PR_KNIGHT_ACTIONS + DROP
    STATE_STACK_HEIGHT = (len(ORDER) + (2 * len(HAND_ORDER)) + ALLOWED_REPEATS + 1 + 1)

    def __init__(self):
        self.reset()

    def reset(self):
        self.game_state = GameState.from_board((Game.INITIAL_LAYOUT, [], [], 0, 'W', 0))
        self.state_history = []

    def take_action(self, action):
        next_state = self.game_state.action_to_state(action)
        self.state_history.append(next_state)
        self.game_state = next_state

    def move_to_next_state(self, next_state):
        self.state_history.append(next_state)
        self.game_state = next_state

    
    @staticmethod
    def action_matrix_to_array(actions):
        action_pool = []
        for y in range(0, Game.BOARD_Y):
            for x in range(0, Game.BOARD_X):
                for z in range(0, len(actions[0][0])):
                    if(actions[z][y][x] == 1):
                        action_pool.append((x, y, z))
        return action_pool

    @staticmethod
    def piece_actions(piece):
        allowed_actions = np.zeros((8, Game.MAX_MOVE_MAGNITUDE), dtype=int)
        if(piece == 'P'):
            allowed_actions[0][0] = 1
        elif(piece == 'S'):
            allowed_actions[0][0] = 1
            allowed_actions[1][0] = 1
            allowed_actions[3][0] = 1
            allowed_actions[5][0] = 1
            allowed_actions[7][0] = 1
        elif(piece == 'G' or piece == '+P' or piece == '+S'):
            allowed_actions[0][0] = 1
            allowed_actions[1][0] = 1
            allowed_actions[2][0] = 1
            allowed_actions[4][0] = 1
            allowed_actions[6][0] = 1
            allowed_actions[7][0] = 1
        elif(piece == 'B'):
            for i in range(0, Game.MAX_MOVE_MAGNITUDE):
                allowed_actions[1][i] = 1
                allowed_actions[3][i] = 1
                allowed_actions[5][i] = 1
                allowed_actions[7][i] = 1
        elif(piece == 'R'):
            for i in range(0, Game.MAX_MOVE_MAGNITUDE):
                allowed_actions[0][i] = 1
                allowed_actions[2][i] = 1
                allowed_actions[4][i] = 1
                allowed_actions[6][i] = 1
        elif(piece == '+B'):
            for i in range(0, Game.MAX_MOVE_MAGNITUDE):
                allowed_actions[1][i] = 1
                allowed_actions[3][i] = 1
                allowed_actions[5][i] = 1
                allowed_actions[7][i] = 1
            allowed_actions[0][0] = 1
            allowed_actions[2][0] = 1
            allowed_actions[4][0] = 1
            allowed_actions[6][0] = 1
        elif(piece == '+R'):
            for i in range(0, Game.MAX_MOVE_MAGNITUDE):
                allowed_actions[0][i] = 1
                allowed_actions[2][i] = 1
                allowed_actions[4][i] = 1
                allowed_actions[6][i] = 1
            allowed_actions[1][0] = 1
            allowed_actions[7][0] = 1
        elif(piece == 'K'):
            for i in range(0, 8):
                allowed_actions[i][0] = 1
        return allowed_actions

    @staticmethod
    def get_coordinates(x, y, magnitude, direction):
        new_x = x
        new_y = y
        if(direction >= 1 and direction <= 3):
            new_x = x + (magnitude)
        if(direction >= 5 and direction <= 7):
            new_x = x - (magnitude)
        if(direction >= 3 and direction <= 5):
            new_y = y + (magnitude)
        if((direction >= 7) or (direction <= 1)):
            new_y = y - (magnitude)
        return (new_x, new_y)

    @staticmethod
    def get_direction(x, y, new_x, new_y):
        if(new_x == x and new_y < y):
            direction = 0
        elif(new_x > x and new_y < y):
            direction = 1
        elif(new_x > x and new_y == y):
            direction = 2
        elif(new_x > x and new_y < y):
            direction = 3
        elif(new_x == x and new_y < y):
            direction = 4
        elif(new_x < x and new_y < y):
            direction = 5
        elif(new_x < x and new_y == y):
            direction = 6
        elif(new_x < x and new_y > y):
            direction = 7
        else:
            raise 'Invalid coordinates!'
        
        magnitude = max(abs(x-new_x), abs(y-new_y))
        if(abs(x-new_x) != 0 and abs(x-new_x) != magnitude and abs(y-new_y) != 0 and abs(y-new_y) != magnitude):
            raise 'Invalid coordinates!'

        return (magnitude, direction)

    @classmethod
    def check_bounds(cls, x, y):
        return x < cls.BOARD_X and x >= 0 and y < cls.BOARD_Y and y >= 0

    @staticmethod
    def is_promoted(piece):
        return piece[0] == '+' or piece == 'K' or piece =='k' or piece =='G' or piece == 'g' 

    @staticmethod
    def unpromote(piece):
        return piece.replace('+', '').upper()

    @staticmethod
    def promote(piece):
        if Game.is_promoted(piece): 
            return piece
        else: 
            return '+' + piece.upper()

    @staticmethod
    def game_reward():
       return 1

class GameState:
    def __init__(self, args):
        (board, hand1, hand2, repetitions, colour, move_count) = args
        self.board = board
        self.hand1 = hand1
        self.hand2 = hand2
        self.repetitions = repetitions
        self.colour = colour
        self.move_count = move_count
        logger.info(self.print())


    @classmethod
    def from_plane_stack(cls,stack):
        return cls(cls.plane_stack_to_board(stack))

    @classmethod
    def from_board(cls, args):
        return cls(args)

    def transition(self):
        self.flip()
        tmp = self.hand1
        self.hand1 = self.hand2
        self.hand2 = tmp
        self.colour = 'W' if (self.colour == 'B') else 'B'
        self.move_count += 1

    def flip(self):
        newboard = [x[:] for x in self.board]

        for x in range(0, Game.BOARD_X):
            for y in range(0, Game.BOARD_Y):
                if(self.board[y][x].lower() == self.board[y][x]):
                    self.board[y][x] = self.board[y][x].upper()
                elif(self.board[y][x].upper() == self.board[y][x]):
                    self.board[y][x] = self.board[y][x].lower()
                newboard[Game.BOARD_Y-y-1][Game.BOARD_X-x-1] = self.board[y][x]
        self.board = newboard

    def allowed_actions_matrix(self):
        actions = np.zeros((Game.ACTION_STACK_HEIGHT, Game.BOARD_Y, Game.BOARD_X), dtype=int)
        #board, hand1, hand2, repetitions, colour, move_count = GameState.plane_stack_to_board(
        #    self.stack)  # TODO REPETITIONS
        #logger.debug('\n' + str(board).replace('], ', ']\n'))
        for x in range(0, Game.BOARD_X):
            for y in range(0, Game.BOARD_Y):
                if(self.board[y][x] in Game.ORDER[0:((len(Game.ORDER)//2))]):  # for all your peices
                    logger.debug(str(self.board[y][x]) +
                                  ' at ' + str(x) + ', '+str(y))
                    piece_possible_actions = Game.piece_actions(self.board[y][x])
                    #logger.debug('\n'+ str(piece_possible_actions))
                    # TODO: add knight actions
                    for direction in range(0, 8):
                        for magnitude in range(1, Game.MAX_MOVE_MAGNITUDE):
                            if piece_possible_actions[direction][magnitude-1]:
                                new_x, new_y = Game.get_coordinates(
                                    x, y, magnitude, direction)
                                if(Game.check_bounds(new_x, new_y)):
                                    if(self.board[new_y][new_x] not in Game.ORDER[0:((len(Game.ORDER)//2))]):
                                        clear = True
                                        for i in range(1, magnitude):
                                            current_x, current_y = Game.get_coordinates(
                                                x, y, i, direction)
                                            #logger.debug('looking at {0}, {1}'.format(current_x, current_y))
                                            if self.board[current_y][current_x] != '.':
                                                clear = False
                                                break

                                        if(clear):
                                            logger.debug('QUEEN MOVE: {0} from ({1}, {2}) to ({3}, {4})'.format(
                                                self.board[y][x], x, y, new_x, new_y))
                                            actions[direction *
                                                          (Game.MAX_MOVE_MAGNITUDE)+(magnitude-1)][y][x] = 1
                                            #logger.debug('z = ' + str(direction *
                                            #              Game.MAX_MOVE_MAGNITUDE+(magnitude-1)))
                                            if(new_y < Game.PROMOTION_ZONE_SIZE or y < Game.PROMOTION_ZONE_SIZE):
                                                if(not Game.is_promoted(self.board[y][x])):
                                                    logger.debug('PROMOTION QUEEN MOVE: {0} from ({1}, {2}) to ({3}, {4})'.format(
                                                        self.board[y][x], x, y, new_x, new_y))
                                                    actions[Game.QUEEN_ACTIONS+ Game.KNIGHT_ACTIONS +
                                                                  direction*(Game.MAX_MOVE_MAGNITUDE)+(magnitude-1)][y][x] = 1
        for dropcount in range(0, Game.DROP):
            if(Game.HAND_ORDER[dropcount] in self.hand1):
                for y in range(0, Game.BOARD_Y):
                    for x in range(0, Game.BOARD_X):
                        if self.board[y][x] == '.':
                            if(Game.HAND_ORDER[dropcount] != 'P' or (y != 0 and self.board[y-1][x] != 'k' and ('P' not in (row[x] for row in self.board)))):
                                logger.debug('DROP: {0} to ({1}, {2})'.format(
                                    Game.HAND_ORDER[dropcount], x, y))
                                actions[Game.QUEEN_ACTIONS + Game.KNIGHT_ACTIONS +
                                                Game.PR_QUEEN_ACTIONS + Game.PR_KNIGHT_ACTIONS+dropcount][y][x] = 1
        return actions

    def next_states_array(self):
        allowed_states = []

        for x in range(0, Game.BOARD_X):
            for y in range(0, Game.BOARD_Y):
                if(self.board[y][x] in Game.ORDER[0:((len(Game.ORDER)//2))]):  # for all your peices
                    logger.debug(str(self.board[y][x]) +
                                  ' at ' + str(x) + ', '+str(y))
                    piece_possible_actions = Game.piece_actions(self.board[y][x])
                    #logger.debug('\n'+ str(piece_possible_actions))
                    # TODO: add knight actions
                    for direction in range(0, 8):
                        for magnitude in range(1, Game.MAX_MOVE_MAGNITUDE):
                            if piece_possible_actions[direction][magnitude-1]:
                                new_x, new_y = Game.get_coordinates(
                                    x, y, magnitude, direction)
                                if(Game.check_bounds(new_x, new_y)):
                                    if(self.board[new_y][new_x] not in Game.ORDER[0:((len(Game.ORDER)//2))]):
                                        clear = True
                                        for i in range(1, magnitude):
                                            current_x, current_y = Game.get_coordinates(
                                                x, y, i, direction)
                                            #logger.debug('looking at {0}, {1}'.format(current_x, current_y))
                                            if self.board[current_y][current_x] != '.':
                                                clear = False
                                                break

                                        if(clear):
                                            next_state= copy.deepcopy(self)
                                            logger.debug('QUEEN MOVE: {0} from ({1}, {2}) to ({3}, {4})'.format(
                                                self.board[y][x], x, y, new_x, new_y))

                                            if self.board[new_y][new_x] != '.' and self.board[new_y][new_x] != 'k':
                                                next_state.hand1.append(Game.unpromote(next_state.board[new_y][new_x]))
                                            next_state.board[new_y][new_x] = next_state.board[y][x]
                                            next_state.board[y][x] = '.'
                                                
                                            next_state.transition()
                                            allowed_states.append(next_state)

                                            if(new_y < Game.PROMOTION_ZONE_SIZE or y < Game.PROMOTION_ZONE_SIZE):
                                                if(not Game.is_promoted(self.board[y][x])):
                                                    logger.debug('PROMOTION QUEEN MOVE: {0} from ({1}, {2}) to ({3}, {4})'.format(
                                                        self.board[y][x], x, y, new_x, new_y))
                                                    next_state= copy.deepcopy(self)

                                                    if self.board[new_y][new_x] != '.' and self.board[new_y][new_x] != 'k':
                                                        next_state.hand1.append(Game.unpromote(next_state.board[new_y][new_x]))
                                                    next_state.board[new_y][new_x] = Game.promote(next_state.board[y][x])
                                                    next_state.board[y][x] = '.'

                                                    next_state.transition()
                                                    allowed_states.append(next_state)

                for dropcount in range(0, Game.DROP):
                    if(Game.HAND_ORDER[dropcount] in self.hand1):
                        if self.board[y][x] == '.':
                            if(Game.HAND_ORDER[dropcount] != 'P') or (y != 0 and self.board[y-1][x] != 'k' and ('P' not in (row[x] for row in self.board))):
                                logger.debug('DROP: {0} to ({1}, {2})'.format(Game.HAND_ORDER[dropcount], x, y))
                                piece = Game.HAND_ORDER[dropcount]
                                next_state= copy.deepcopy(self)
                                next_state.board[y][x] = piece
                                next_state.hand1.remove(piece)                                
                                next_state.transition()
                                allowed_states.append(next_state)
        return allowed_states

    def action_to_state(self, action):
        next_state= copy.deepcopy(self)
        (x, y, z) = action
        if(z < Game.QUEEN_ACTIONS):
            direction = z // Game.MAX_MOVE_MAGNITUDE
            magnitude = (z % Game.MAX_MOVE_MAGNITUDE) + 1
            new_x, new_y = Game.get_coordinates(x, y, magnitude, direction)
            if next_state.board[new_y][new_x] != '.' and next_state.board[new_y][new_x] != 'k':
                next_state.hand1.append(Game.unpromote(next_state.board[new_y][new_x]))
            next_state.board[new_y][new_x] = next_state.board[y][x]
            next_state.board[y][x] = '.'
        elif z < Game.QUEEN_ACTIONS + Game.KNIGHT_ACTIONS:  # TODO knight moves
            pass
        elif z < Game.QUEEN_ACTIONS + Game.KNIGHT_ACTIONS + Game.PR_QUEEN_ACTIONS:
            direction = (z- Game.QUEEN_ACTIONS -
                         Game.KNIGHT_ACTIONS) // Game.MAX_MOVE_MAGNITUDE
            magnitude = ((z - Game.QUEEN_ACTIONS - Game.KNIGHT_ACTIONS) %
                         Game.MAX_MOVE_MAGNITUDE) + 1
            new_x, new_y = Game.get_coordinates(x, y, magnitude, direction)
            if next_state.board[new_y][new_x] != '.' and next_state.board[new_y][new_x] != 'k':
                next_state.hand1.append(Game.unpromote(next_state.board[new_y][new_x]))
            next_state.board[new_y][new_x] = Game.promote(next_state.board[y][x])
            next_state.board[y][x] = '.'
        elif z < Game.QUEEN_ACTIONS + Game.KNIGHT_ACTIONS + Game.PR_QUEEN_ACTIONS + Game.PR_KNIGHT_ACTIONS:
            pass #TODO PROMOTED KINGHT MOVES
        else:
            piece_index = z- Game.QUEEN_ACTIONS - Game.KNIGHT_ACTIONS - Game.PR_KNIGHT_ACTIONS - Game.PR_QUEEN_ACTIONS
            piece = Game.HAND_ORDER[piece_index]
            next_state.board[y][x] = piece
            #logger.debug('piece = ' + next_state.board[y][x])            
            #logger.debug(next_state.hand1)
            next_state.hand1.remove(piece)
        next_state.flip()
        #logger.debug('\n' + str(self.board).replace('], ', ']\n'))
        tmp = next_state.hand1
        next_state.hand1 = next_state.hand2
        next_state.hand2 = tmp
        next_state.colour = 'W' if (next_state.colour == 'B') else 'B'
        next_state.move_count += 1
        #self.stack = GameState.board_to_plane_stack(
        #    board, hand1, hand2, repetitions, colour, move_count)
        return next_state


    def game_ended(self):
        #return not ((np.any(self.stack[Game.ORDER.index("K")])) and (np.any(self.stack[Game.ORDER.index('k')])))
        return not(any('K' in sublist for sublist in self.board) and any('k' in sublist for sublist in self.board)) 

    @staticmethod
    def board_to_plane_stack(board, hand1, hand2, repetitions, colour, move_count):
        stack = np.zeros((Game.STATE_STACK_HEIGHT, Game.BOARD_Y, Game.BOARD_X), dtype=int)
        for y in range(0, len(board)):
            for x in range(0, len(board[0])):
                if(board[y][x] != '.'):
                    stack[Game.ORDER.index(board[y][x])][y][x] = 1
        for i in range(0, len(hand1)):
            stack[len(Game.ORDER) +
                  Game.HAND_ORDER.index(hand1[i].upper())] += 1
        for i in range(0, len(hand2)):
            stack[len(Game.ORDER) + len(Game.HAND_ORDER) +
                  Game.HAND_ORDER.index(hand2[i].upper())] += 1
        for i in range(0, repetitions):
            stack[len(Game.ORDER) + (2 * len(Game.HAND_ORDER)) +
                  i] = np.ones((Game.BOARD_X, Game.BOARD_Y), dtype=int)
        stack[len(Game.ORDER) + (2 * len(Game.HAND_ORDER)) +
              Game.ALLOWED_REPEATS] = (np.ones((Game.BOARD_X, Game.BOARD_Y), dtype=int) if colour == 'W' else np.ones((Game.BOARD_X, Game.BOARD_Y), dtype=int)*-1)
        stack[len(Game.ORDER) + (2 * len(Game.HAND_ORDER)) +
              Game.ALLOWED_REPEATS + 1] = np.ones((Game.BOARD_X, Game.BOARD_Y), dtype=int) * move_count
        return stack

    """
    convert a stack of planes to a human-friendly representation
    multiplies each layer by a number, then adds them up
    """
    @staticmethod
    def plane_stack_to_board(stack):
        board = [['.' for i in range(Game.BOARD_X)]
                 for j in range(Game.BOARD_Y)]
        stack_copy = np.copy(stack)
        for i in range(0, len(Game.ORDER)):
            stack_copy[i] = np.multiply(stack_copy[i], i+1)
        stack_copy = np.sum(stack_copy[0:len(Game.ORDER)], axis=0)
        for y in range(0, stack_copy.shape[0]):
            for x in range(0, stack_copy.shape[1]):
                if stack_copy[y][x] != 0:
                    board[y][x] = Game.ORDER[stack_copy[y][x]-1]
        hand1 = []
        for i in range(len(Game.ORDER), len(Game.ORDER) + len(Game.HAND_ORDER)):
            for _ in range(0, stack[i][0][0]):
                hand1.append(Game.HAND_ORDER[i - len(Game.ORDER)])

        hand2 = []

        for i in range(len(Game.ORDER) + len(Game.HAND_ORDER), len(Game.ORDER) + (2 * len(Game.HAND_ORDER))):
            for _ in range(0, stack[i][0][0]):
                hand2.append(Game.HAND_ORDER[i - len(Game.ORDER)].lower)

        repetitions = 0

        for i in range(len(Game.ORDER) + (2 * len(Game.HAND_ORDER)), len(Game.ORDER) + (2 * len(Game.HAND_ORDER)) + Game.ALLOWED_REPEATS):
            if stack[i][0][0] == 1:
                repetitions += 1
        # todo - change [0][0] to all
        colour = 'W' if stack[len(
            Game.ORDER) + (2 * len(Game.HAND_ORDER)) + Game.ALLOWED_REPEATS][0][0] == 1 else 'B'
        move_count = stack[len(
            Game.ORDER) + (2 * len(Game.HAND_ORDER)) + Game.ALLOWED_REPEATS + 1][0][0]

        return (board, hand1, hand2, repetitions, colour, move_count)

    def print(self, level=0, flip=False):
        margin = '\t' * level
        state_copy = deepcopy(self)
        if flip:
            state_copy.flip()    
        return '\n{0}--{1}--\n{0}{2}\n{0}Hand1: {3}\n{0}Hand2: {4}\n{0}Colour: {5}'.format(margin, self.move_count, str(state_copy.board).replace('], ', ']\n'+ margin), self.hand1, self.hand2, self.colour)
        
        #return '\n--'+ str(self.move_count) +'--\n' + str(self.board).replace('], ', ']\n')+'\nHand1: ' + str(self.hand1) + '\nHand2: ' + str(self.hand2) + '\nColour: ' + str(self.colour)

    def __hash__(self):
        return hash(str(GameState.board_to_plane_stack(self.board, self.hand1, self.hand2, self.repetitions, self.colour, self.move_count)))

    def compare_boards(self, other): 
        return self.board==other.board and Counter(self.hand1) == Counter(other.hand1) and Counter(self.hand2) == Counter(other.hand2) and self.colour == other.colour
        #TODO: test?

    def __eq__(self, other):
        return hash(str(GameState.board_to_plane_stack(self.board, self.hand1, self.hand2, self.repetitions, self.colour, self.move_count))) == hash(str(GameState.board_to_plane_stack(other.board, other.hand1, other.hand2, other.repetitions, other.colour, other.move_count)))
       # return hash(str(GameState.board_to_plane_stack(self.board, self.hand1, self.hand2, self.repetitions, self.colour, self.move_count)) == hash(str(GameState.board_to_plane_stack(other.board, other.hand1, other.hand2, other.repetitions, other.colour, other.move_count))))

    def __ne__(self, other): 
        return not (self == other)

    