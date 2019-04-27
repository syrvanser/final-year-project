import numpy as np

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

    MAX_MOVE_MAGNITUDE = (max(BOARD_X - 1, BOARD_Y - 1))
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
        from games import MiniShogiGameState  # circular dependency - needs to be fixed later
        self.state_history = []

        self.game_state = MiniShogiGameState.from_board((MiniShogiGame.INITIAL_LAYOUT, [], [], 0, 'W', 0))
        self.state_history.append(self.game_state)

    def take_action(self, action):

        from games import MiniShogiGameState  # circular dependency - needs to be fixed later

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
    def get_direction(x, y, new_x, new_y):  # 2 4 1 3
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
