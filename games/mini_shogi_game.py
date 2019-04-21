import numpy as np
from collections import Counter
import copy
from copy import deepcopy
import logging

from games import Game


class MiniShogiGame(Game):
    BOARD_X = 5
    BOARD_Y = 5
    ALLOWED_REPEATS = 0
    INITIAL_LAYOUT = [['r', 'b', 's', 'g', 'k'],
                      ['.', '.', '.', '.', 'p'],
                      ['.', '.', '.', '.', '.'],
                      ['P', '.', '.', '.', '.'],
                      ['K', 'G', 'S', 'B', 'R']]

    ORDER = ['P', 'S', 'G', 'B', 'R', '+P', '+S', '+B', '+R',
             'K', 'p', 's', 'g', 'b', 'r', '+p', '+s', '+b', '+r', 'k']
    HAND_ORDER = ['P', 'S', 'G', 'B', 'R']

    MAX_MOVE_MAGNITUDE = (max(BOARD_X-1, BOARD_Y-1))
    QUEEN_ACTIONS = 8 * MAX_MOVE_MAGNITUDE
    KNIGHT_ACTIONS = 0  # no knights in minishogi
    PR_QUEEN_ACTIONS = QUEEN_ACTIONS
    PR_KNIGHT_ACTIONS = KNIGHT_ACTIONS
    PROMOTION_ZONE_SIZE = 1
    DROP = len(HAND_ORDER)
    ACTION_STACK_HEIGHT = QUEEN_ACTIONS + KNIGHT_ACTIONS + PR_QUEEN_ACTIONS + PR_KNIGHT_ACTIONS + DROP
    STATE_STACK_HEIGHT = (len(ORDER) + (2 * len(HAND_ORDER)) + ALLOWED_REPEATS + 1 + 1)

    def __init__(self):
        super().__init__()
        self.state_history = []
        self.game_state = MiniShogiGameState.from_board((MiniShogiGame.INITIAL_LAYOUT, [], [], 0, 'W', 0))
        self.state_history.append(self.game_state)

    def take_action(self, action):
        next_state = MiniShogiGameState.action_to_state(self.game_state, action)
        self.state_history.append(next_state)
        self.game_state = next_state

    def move_to_next_state(self, next_state):
        self.state_history.append(next_state)
        self.game_state = next_state

    @classmethod
    def piece_actions(cls, piece):
        allowed_actions = np.zeros((8, cls.MAX_MOVE_MAGNITUDE), dtype=int)
        if piece == 'P':
            allowed_actions[0][0] = 1
        elif piece == 'S':
            allowed_actions[0][0] = 1
            allowed_actions[1][0] = 1
            allowed_actions[3][0] = 1
            allowed_actions[5][0] = 1
            allowed_actions[7][0] = 1
        elif piece == 'G' or piece == '+P' or piece == '+S':
            allowed_actions[0][0] = 1
            allowed_actions[1][0] = 1
            allowed_actions[2][0] = 1
            allowed_actions[4][0] = 1
            allowed_actions[6][0] = 1
            allowed_actions[7][0] = 1
        elif piece == 'B':
            for i in range(0, cls.MAX_MOVE_MAGNITUDE):
                allowed_actions[1][i] = 1
                allowed_actions[3][i] = 1
                allowed_actions[5][i] = 1
                allowed_actions[7][i] = 1
        elif piece == 'R':
            for i in range(0, cls.MAX_MOVE_MAGNITUDE):
                allowed_actions[0][i] = 1
                allowed_actions[2][i] = 1
                allowed_actions[4][i] = 1
                allowed_actions[6][i] = 1
        elif piece == '+B':
            for i in range(0, cls.MAX_MOVE_MAGNITUDE):
                allowed_actions[1][i] = 1
                allowed_actions[3][i] = 1
                allowed_actions[5][i] = 1
                allowed_actions[7][i] = 1
            allowed_actions[0][0] = 1
            allowed_actions[2][0] = 1
            allowed_actions[4][0] = 1
            allowed_actions[6][0] = 1
        elif piece == '+R':
            for i in range(0, cls.MAX_MOVE_MAGNITUDE):
                allowed_actions[0][i] = 1
                allowed_actions[2][i] = 1
                allowed_actions[4][i] = 1
                allowed_actions[6][i] = 1
            allowed_actions[1][0] = 1
            allowed_actions[7][0] = 1
        elif piece == 'K':
            for i in range(0, 8):
                allowed_actions[i][0] = 1
        return allowed_actions

    @staticmethod
    def get_coordinates(x, y, magnitude, direction):
        new_x = x
        new_y = y
        if 1 <= direction <= 3:
            new_x = x + magnitude
        if 5 <= direction <= 7:
            new_x = x - magnitude
        if 3 <= direction <= 5:
            new_y = y + magnitude
        if (direction >= 7) or (direction <= 1):
            new_y = y - magnitude
        return new_x, new_y

    @staticmethod
    def get_direction(x, y, new_x, new_y): #2 4 1 3
        if new_x == x and new_y < y:
            direction = 0
        elif new_x > x and new_y < y:
            direction = 1
        elif new_x > x and new_y == y:
            direction = 2
        elif new_x > x and new_y > y:
            direction = 3
        elif new_x == x and new_y > y:
            direction = 4
        elif new_x < x and new_y > y:
            direction = 5
        elif new_x < x and new_y == y:
            direction = 6
        elif new_x < x and new_y < y:
            direction = 7
        else:
            raise Exception('Invalid coordinates!')

        magnitude = max(abs(x - new_x), abs(y - new_y))
        if (abs(x - new_x) != 0 and abs(x - new_x) != magnitude and abs(y - new_y) != 0 and abs(
                y - new_y) != magnitude):
            raise Exception('Invalid coordinates!')

        return magnitude, direction

    @classmethod
    def check_bounds(cls, x, y):
        return cls.BOARD_X > x >= 0 and cls.BOARD_Y > y >= 0

    @staticmethod
    def is_promoted(piece):
        return piece[0] == '+' or piece == 'K' or piece == 'k' or piece == 'G' or piece == 'g'

    @classmethod
    def demote(cls, piece):
        return piece.replace('+', '').upper()

    @classmethod
    def promote(cls, piece):
        if cls.is_promoted(piece):
            return piece
        else:
            return '+' + piece.upper()


class MiniShogiGameState:
    def __init__(self, args):
        (board, hand1, hand2, repetitions, colour, move_count) = args
        self.board = board
        self.hand1 = hand1
        self.hand2 = hand2
        self.repetitions = repetitions
        self.colour = colour
        self.move_count = move_count
        self.hash = hash((self.state_to_plane_stack(self)).tostring())

    @classmethod
    def from_plane_stack(cls, stack):
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
        self.hash = hash((self.state_to_plane_stack(self)).tostring())

    def flip(self):
        newboard = [x[:] for x in self.board]

        for x in range(0, MiniShogiGame.BOARD_X):
            for y in range(0, MiniShogiGame.BOARD_Y):
                if self.board[y][x].lower() == self.board[y][x]:
                    self.board[y][x] = self.board[y][x].upper()
                elif self.board[y][x].upper() == self.board[y][x]:
                    self.board[y][x] = self.board[y][x].lower()
                newboard[MiniShogiGame.BOARD_Y - y - 1][MiniShogiGame.BOARD_X - x - 1] = self.board[y][x]
        self.board = newboard
        self.hash = hash((self.state_to_plane_stack(self)).tostring())

    def allowed_actions_matrix(self):
        actions = np.zeros((MiniShogiGame.ACTION_STACK_HEIGHT, MiniShogiGame.BOARD_Y, MiniShogiGame.BOARD_X), dtype=int)
        # TODO REPETITIONS
        for x in range(0, MiniShogiGame.BOARD_X):
            for y in range(0, MiniShogiGame.BOARD_Y):
                if (self.board[y][x] in MiniShogiGame.ORDER[
                                        0:(len(MiniShogiGame.ORDER) // 2)]):  # for all your peices
                    # logging.debug(str(self.board[y][x]) + ' at ' + str(x) + ', ' + str(y))
                    piece_possible_actions = MiniShogiGame.piece_actions(self.board[y][x])
                    # logging.debug('\n'+ str(piece_possible_actions))
                    # TODO: add knight actions
                    for direction in range(0, 8):
                        for magnitude in range(1, MiniShogiGame.MAX_MOVE_MAGNITUDE+1):
                            if piece_possible_actions[direction][magnitude - 1]:
                                new_x, new_y = MiniShogiGame.get_coordinates(
                                    x, y, magnitude, direction)
                                if MiniShogiGame.check_bounds(new_x, new_y):
                                    if (self.board[new_y][new_x] not in MiniShogiGame.ORDER[
                                                                        0:(len(MiniShogiGame.ORDER) // 2)]):
                                        clear = True
                                        for i in range(1, magnitude):
                                            current_x, current_y = MiniShogiGame.get_coordinates(
                                                x, y, i, direction)
                                            if self.board[current_y][current_x] != '.':
                                                clear = False
                                                break

                                        if clear:
                                            # logging.debug('QUEEN MOVE: {0} from ({1}, {2}) to ({3}, {4})'.format(
                                            # self.board[y][x], x, y, new_x, new_y))
                                            if (self.board[x][y] != 'P') or (new_y < MiniShogiGame.BOARD_Y - 1): #autopromote pawns
                                                actions[direction *
                                                        MiniShogiGame.MAX_MOVE_MAGNITUDE + (magnitude - 1)][y][x] = 1

                                            if (
                                                    new_y < MiniShogiGame.PROMOTION_ZONE_SIZE or y < MiniShogiGame.PROMOTION_ZONE_SIZE):
                                                if not MiniShogiGame.is_promoted(self.board[y][x]):
                                                    # logging.debug(
                                                    #   'PROMOTION QUEEN MOVE: {0} from ({1}, {2}) to ({3}, {4})'.format(
                                                    #      self.board[y][x], x, y, new_x, new_y))
                                                    actions[MiniShogiGame.QUEEN_ACTIONS + MiniShogiGame.KNIGHT_ACTIONS +
                                                            direction * MiniShogiGame.MAX_MOVE_MAGNITUDE + (
                                                                    magnitude - 1)][y][x] = 1
        for dropcount in range(0, MiniShogiGame.DROP):
            if MiniShogiGame.HAND_ORDER[dropcount] in self.hand1:
                for y in range(0, MiniShogiGame.BOARD_Y):
                    for x in range(0, MiniShogiGame.BOARD_X):
                        if self.board[y][x] == '.':
                            if (MiniShogiGame.HAND_ORDER[dropcount] != 'P' or (
                                    y != 0 and self.board[y - 1][x] != 'k' and (
                                    'P' not in (row[x] for row in self.board)))):
                                # logging.debug('DROP: {0} to ({1}, {2})'.format(
                                #   MiniShogiGame.HAND_ORDER[dropcount], x, y))
                                actions[MiniShogiGame.QUEEN_ACTIONS + MiniShogiGame.KNIGHT_ACTIONS +
                                        MiniShogiGame.PR_QUEEN_ACTIONS + MiniShogiGame.PR_KNIGHT_ACTIONS + dropcount][
                                    y][x] = 1
        return actions

    def next_states_array(self):
        allowed_states = []

        for x in range(0, MiniShogiGame.BOARD_X):
            for y in range(0, MiniShogiGame.BOARD_Y):
                if (self.board[y][x] in MiniShogiGame.ORDER[
                                        0:(len(MiniShogiGame.ORDER) // 2)]):  # for all your pieces
                    # logging.debug(str(self.board[y][x]) +
                    #             ' at ' + str(x) + ', ' + str(y))
                    piece_possible_actions = MiniShogiGame.piece_actions(self.board[y][x])
                    # logging.debug('\n'+ str(piece_possible_actions))
                    # TODO: add knight actions
                    for direction in range(0, 8):
                        for magnitude in range(1, MiniShogiGame.MAX_MOVE_MAGNITUDE):
                            if piece_possible_actions[direction][magnitude - 1]:
                                new_x, new_y = MiniShogiGame.get_coordinates(
                                    x, y, magnitude, direction)
                                if MiniShogiGame.check_bounds(new_x, new_y):
                                    if (self.board[new_y][new_x] not in MiniShogiGame.ORDER[
                                                                        0:(len(MiniShogiGame.ORDER) // 2)]):
                                        clear = True
                                        for i in range(1, magnitude):
                                            current_x, current_y = MiniShogiGame.get_coordinates(
                                                x, y, i, direction)
                                            # logging.debug('looking at {0}, {1}'.format(current_x, current_y))
                                            if self.board[current_y][current_x] != '.':
                                                clear = False
                                                break

                                        if clear:
                                            next_state = copy.deepcopy(self)
                                            # logging.debug('QUEEN MOVE: {0} from ({1}, {2}) to ({3}, {4})'.format(
                                            #    self.board[y][x], x, y, new_x, new_y))

                                            if self.board[new_y][new_x] != '.' and self.board[new_y][new_x] != 'k':
                                                next_state.hand1.append(
                                                    MiniShogiGame.demote(next_state.board[new_y][new_x]))
                                            next_state.board[new_y][new_x] = next_state.board[y][x]
                                            next_state.board[y][x] = '.'

                                            next_state.transition()
                                            allowed_states.append(next_state)

                                            if (
                                                    new_y < MiniShogiGame.PROMOTION_ZONE_SIZE or y < MiniShogiGame.PROMOTION_ZONE_SIZE):
                                                if not MiniShogiGame.is_promoted(self.board[y][x]):
                                                    # logging.debug(
                                                    #    'PROMOTION QUEEN MOVE: {0} from ({1}, {2}) to ({3}, {4})'.format(
                                                    #        self.board[y][x], x, y, new_x, new_y))
                                                    next_state = copy.deepcopy(self)

                                                    if self.board[new_y][new_x] != '.' and self.board[new_y][
                                                        new_x] != 'k':
                                                        next_state.hand1.append(
                                                            MiniShogiGame.demote(next_state.board[new_y][new_x]))
                                                    next_state.board[new_y][new_x] = MiniShogiGame.promote(
                                                        next_state.board[y][x])
                                                    next_state.board[y][x] = '.'

                                                    next_state.transition()
                                                    allowed_states.append(next_state)

                for dropcount in range(0, MiniShogiGame.DROP):
                    if MiniShogiGame.HAND_ORDER[dropcount] in self.hand1:
                        if self.board[y][x] == '.':
                            if (MiniShogiGame.HAND_ORDER[dropcount] != 'P') or (
                                    y != 0 and self.board[y - 1][x] != 'k' and (
                                    'P' not in (row[x] for row in self.board))):
                                # logging.debug(
                                #    'DROP: {0} to ({1}, {2})'.format(MiniShogiGame.HAND_ORDER[dropcount], x, y))
                                piece = MiniShogiGame.HAND_ORDER[dropcount]
                                next_state = copy.deepcopy(self)
                                next_state.board[y][x] = piece
                                next_state.hand1.remove(piece)
                                next_state.transition()
                                allowed_states.append(next_state)
        return allowed_states

    @staticmethod
    def action_matrix_to_action_array(matrix):
        array = []
        for x in range(0, MiniShogiGame.BOARD_X):
            for y in range(0, MiniShogiGame.BOARD_Y):
                for z in range(0, MiniShogiGame.ACTION_STACK_HEIGHT):
                    if matrix[z][y][x] != 0:
                        array.append((z, y, x))
        return array

    @staticmethod
    def action_matrix_to_state_array(state, matrix):
        array = []
        for x in range(0, MiniShogiGame.BOARD_X):
            for y in range(0, MiniShogiGame.BOARD_Y):
                for z in range(0, MiniShogiGame.ACTION_STACK_HEIGHT):
                    if matrix[z][y][x] != 0:
                        array.append(MiniShogiGameState.action_to_state(state, (z, y, x)))
        return array

    @staticmethod
    def prob_list_to_matrix(pi_list, actions):
        matrix = np.zeros((MiniShogiGame.ACTION_STACK_HEIGHT, MiniShogiGame.BOARD_Y, MiniShogiGame.BOARD_X))
        for i, a in enumerate(actions):
            matrix[a] = pi_list[i]
        return matrix

    @staticmethod
    def action_to_state(state, action):
        next_state = copy.deepcopy(state)
        (z, y, x) = action #84 2 0
        if z < MiniShogiGame.QUEEN_ACTIONS:
            direction = z // MiniShogiGame.MAX_MOVE_MAGNITUDE
            magnitude = (z % MiniShogiGame.MAX_MOVE_MAGNITUDE) + 1
            new_x, new_y = MiniShogiGame.get_coordinates(x, y, magnitude, direction)
            if next_state.board[new_y][new_x] != '.' and next_state.board[new_y][new_x] != 'k':
                next_state.hand1.append(MiniShogiGame.demote(next_state.board[new_y][new_x]))
            next_state.board[new_y][new_x] = next_state.board[y][x]
            next_state.board[y][x] = '.'
        elif z < MiniShogiGame.QUEEN_ACTIONS + MiniShogiGame.KNIGHT_ACTIONS:  # TODO knight moves
            pass
        elif z < MiniShogiGame.QUEEN_ACTIONS + MiniShogiGame.KNIGHT_ACTIONS + MiniShogiGame.PR_QUEEN_ACTIONS:
            direction = (z - MiniShogiGame.QUEEN_ACTIONS -
                         MiniShogiGame.KNIGHT_ACTIONS) // MiniShogiGame.MAX_MOVE_MAGNITUDE
            magnitude = ((z - MiniShogiGame.QUEEN_ACTIONS - MiniShogiGame.KNIGHT_ACTIONS) %
                         MiniShogiGame.MAX_MOVE_MAGNITUDE) + 1
            new_x, new_y = MiniShogiGame.get_coordinates(x, y, magnitude, direction)
            if next_state.board[new_y][new_x] != '.' and next_state.board[new_y][new_x] != 'k':
                next_state.hand1.append(MiniShogiGame.demote(next_state.board[new_y][new_x]))
            next_state.board[new_y][new_x] = MiniShogiGame.promote(next_state.board[y][x])
            next_state.board[y][x] = '.'
        elif z < MiniShogiGame.QUEEN_ACTIONS + MiniShogiGame.KNIGHT_ACTIONS + MiniShogiGame.PR_QUEEN_ACTIONS + MiniShogiGame.PR_KNIGHT_ACTIONS:
            pass  # TODO PROMOTED KINGHT MOVES
        else:
            piece_index = z - MiniShogiGame.QUEEN_ACTIONS - MiniShogiGame.KNIGHT_ACTIONS - MiniShogiGame.PR_KNIGHT_ACTIONS - MiniShogiGame.PR_QUEEN_ACTIONS
            piece = MiniShogiGame.HAND_ORDER[piece_index]
            next_state.board[y][x] = piece
            next_state.hand1.remove(piece)
        next_state.flip()
        tmp = next_state.hand1
        next_state.hand1 = next_state.hand2
        next_state.hand2 = tmp
        next_state.colour = 'W' if (next_state.colour == 'B') else 'B'
        next_state.move_count += 1
        next_state.hash = hash((next_state.state_to_plane_stack(next_state)).tostring())
        return next_state

    def game_ended(self):
        return not (any('K' in sublist for sublist in self.board) and any('k' in sublist for sublist in self.board))

    """
    @staticmethod
    def board_to_plane_stack(board, hand1, hand2, repetitions, colour, move_count):
        stack = np.zeros((MiniShogiGame.STATE_STACK_HEIGHT, MiniShogiGame.BOARD_Y, MiniShogiGame.BOARD_X), dtype=int)
        for y in range(0, len(board)):
            for x in range(0, len(board[0])):
                if board[y][x] != '.':
                    stack[MiniShogiGame.ORDER.index(board[y][x])][y][x] = 1
        for i in range(0, len(hand1)):
            stack[len(MiniShogiGame.ORDER) +
                  MiniShogiGame.HAND_ORDER.index(hand1[i].upper())] += 1
        for i in range(0, len(hand2)):
            stack[len(MiniShogiGame.ORDER) + len(MiniShogiGame.HAND_ORDER) +
                  MiniShogiGame.HAND_ORDER.index(hand2[i].upper())] += 1
        for i in range(0, repetitions):
            stack[len(MiniShogiGame.ORDER) + (2 * len(MiniShogiGame.HAND_ORDER)) +
                  i] = np.ones((MiniShogiGame.BOARD_X, MiniShogiGame.BOARD_Y), dtype=int)
        stack[len(MiniShogiGame.ORDER) + (2 * len(MiniShogiGame.HAND_ORDER)) +
              MiniShogiGame.ALLOWED_REPEATS] = (
            np.ones((MiniShogiGame.BOARD_X, MiniShogiGame.BOARD_Y), dtype=int) if colour == 'W' else np.ones(
                (MiniShogiGame.BOARD_X, MiniShogiGame.BOARD_Y), dtype=int) * -1)
        stack[len(MiniShogiGame.ORDER) + (2 * len(MiniShogiGame.HAND_ORDER)) +
              MiniShogiGame.ALLOWED_REPEATS + 1] = np.ones((MiniShogiGame.BOARD_X, MiniShogiGame.BOARD_Y),
                                                           dtype=int) * move_count
        return stack
    """

    @staticmethod
    def state_to_plane_stack(state):
        stack = np.zeros((MiniShogiGame.STATE_STACK_HEIGHT, MiniShogiGame.BOARD_Y, MiniShogiGame.BOARD_X), dtype=int)
        for y in range(0, len(state.board)):
            for x in range(0, len(state.board[0])):
                if state.board[y][x] != '.':
                    stack[MiniShogiGame.ORDER.index(state.board[y][x])][y][x] = 1
        for i in range(0, len(state.hand1)):
            stack[len(MiniShogiGame.ORDER) +
                  MiniShogiGame.HAND_ORDER.index(state.hand1[i].upper())] += 1
        for i in range(0, len(state.hand2)):
            stack[len(MiniShogiGame.ORDER) + len(MiniShogiGame.HAND_ORDER) +
                  MiniShogiGame.HAND_ORDER.index(state.hand2[i].upper())] += 1
        '''for i in range(0, state.repetitions):
            stack[len(MiniShogiGame.ORDER) + (2 * len(MiniShogiGame.HAND_ORDER)) +
                  i] = np.ones((MiniShogiGame.BOARD_X, MiniShogiGame.BOARD_Y), dtype=int)'''
        stack[len(MiniShogiGame.ORDER) + (2 * len(MiniShogiGame.HAND_ORDER)) +
              MiniShogiGame.ALLOWED_REPEATS] = (
            np.ones((MiniShogiGame.BOARD_X, MiniShogiGame.BOARD_Y), dtype=int) if state.colour == 'W' else np.ones(
                (MiniShogiGame.BOARD_X, MiniShogiGame.BOARD_Y), dtype=int) * -1)
        stack[len(MiniShogiGame.ORDER) + (2 * len(MiniShogiGame.HAND_ORDER)) +
              MiniShogiGame.ALLOWED_REPEATS + 1] = np.ones((MiniShogiGame.BOARD_X, MiniShogiGame.BOARD_Y),
                                                           dtype=int) * state.move_count
        return stack

    """
    convert a stack of planes to a human-friendly representation
    multiplies each layer by a number, then adds them up
    """

    @staticmethod
    def plane_stack_to_board(stack):
        board = [['.' for _ in range(MiniShogiGame.BOARD_X)]
                 for __ in range(MiniShogiGame.BOARD_Y)]
        stack_copy = np.copy(stack)
        for i in range(0, len(MiniShogiGame.ORDER)):
            stack_copy[i] = np.multiply(stack_copy[i], i + 1)
        stack_copy = np.sum(stack_copy[0:len(MiniShogiGame.ORDER)], axis=0)
        for y in range(0, stack_copy.shape[0]):
            for x in range(0, stack_copy.shape[1]):
                if stack_copy[y][x] != 0:
                    board[y][x] = MiniShogiGame.ORDER[stack_copy[y][x] - 1]
        hand1 = []
        for i in range(len(MiniShogiGame.ORDER), len(MiniShogiGame.ORDER) + len(MiniShogiGame.HAND_ORDER)):
            for _ in range(0, stack[i][0][0]):
                hand1.append(MiniShogiGame.HAND_ORDER[i - len(MiniShogiGame.ORDER)])

        hand2 = []

        for i in range(len(MiniShogiGame.ORDER) + len(MiniShogiGame.HAND_ORDER),
                       len(MiniShogiGame.ORDER) + (2 * len(MiniShogiGame.HAND_ORDER))):
            for _ in range(0, stack[i][0][0]):
                hand2.append(MiniShogiGame.HAND_ORDER[i - len(MiniShogiGame.ORDER)].lower)

        repetitions = 0

        '''for i in range(len(MiniShogiGame.ORDER) + (2 * len(MiniShogiGame.HAND_ORDER)),
                       len(MiniShogiGame.ORDER) + (2 * len(MiniShogiGame.HAND_ORDER)) + MiniShogiGame.ALLOWED_REPEATS):
            if stack[i][0][0] == 1:
                repetitions += 1'''
        colour = 'W' if stack[len(
            MiniShogiGame.ORDER) + (2 * len(MiniShogiGame.HAND_ORDER)) + MiniShogiGame.ALLOWED_REPEATS][0][
                            0] == 1 else 'B'
        move_count = stack[len(
            MiniShogiGame.ORDER) + (2 * len(MiniShogiGame.HAND_ORDER)) + MiniShogiGame.ALLOWED_REPEATS + 1][0][0]

        return board, hand1, hand2, repetitions, colour, move_count

    def print_state(self, level=0, flip=False):
        margin = '\t' * level
        state_copy = deepcopy(self)
        if flip:
            state_copy.flip()
        return '\n{0}--{1}--\n{0}{2}\n{0}Hand1: {3}\n{0}Hand2: {4}\n{0}Colour: {5}'.format(margin, self.move_count, str(
            state_copy.board).replace('], ', ']\n' + margin), self.hand1, self.hand2, self.colour)

    def __hash__(self):
        return self.hash

    def compare_boards(self, other):
        return self.board == other.board and Counter(self.hand1) == Counter(other.hand1) and Counter(
            self.hand2) == Counter(other.hand2) and self.colour == other.colour

    def __eq__(self, other):
        return self.hash == other.hash

    def __ne__(self, other):
        return not (self == other)
