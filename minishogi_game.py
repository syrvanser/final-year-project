import numpy as np
from bidict import bidict


class Game:

    BOARD_X = 5
    BOARD_Y = 5
    ALLOWED_REPEATS = 3
    INITIAL_LAYOUT = [['r', 'b', 's', 'g', 'k'],
            ['', '', '', '', 'p'],
            ['', '', '', '', ''],
            ['P', '', '', '', ''],
            ['K', 'G', 'S', 'B', 'R']]

    """
    dictionary = bidict({'pawn': 'P', 'rook': 'R', 'bishop': 'B', 'lance': 'L', 'silver_general': 'S', 'golden_general': 'G', 'kight': 'N',
                         'king': 'K', 'horse': '+H', 'dragon': '+B', 'tokin': '+P', 'promoted_silver': '+S',
                         'promoted_lance': '+L', 'promoted_knight': '+N'})
    """

    ORDER = ['P', 'S', 'G', 'B', 'R', '+P', '+S', '+B', '+R',
        'K', 'p', 's', 'g', 'b', 'r', '+p', '+s', '+b', '+r', 'k']
    HAND_ORDER = ['P', 'S', 'G', 'B', 'R']

    QUEEN_MOVES = 8 * (max(BOARD_X, BOARD_Y) - 1) 
    KNIGHT_MOVES = 2
    PR_QUEEN_MOVES = QUEEN_MOVES
    PR_KNIGHT_MOVES = KNIGHT_MOVES
    DROP = len(HAND_ORDER)

    def __init__(self):
            self.reset()

    def reset(self):
        self.gameState = GameState(GameState.board_to_plane_stack(
            Game.INITIAL_LAYOUT, [], [], 0, 1, 1))
        self.currentPlayer = 1

    """
    def step(self, action):
            next_state, value, done=self.gameState.takeAction(action)
            self.gameState=next_state
            self.currentPlayer=-self.currentPlayer
            info=None
            return ((next_state, value, done, info))
    """


class GameState:
    def __init__(self, state):
        self.state = state

    def _allowedActions(self):
        moves = np.zeros(Game.QUEEN_MOVES + Game.KNIGHT_MOVES + Game.PR_QUEEN_MOVES + Game.PR_KNIGHT_MOVES + Game.DROP, Game.BOARD_Y, Game.BOARD_X), dtype=int)
        
        return allowed

    """
	def _checkForEndGame(self):

		return 0



	def takeAction(self, action):
		newBoard=np.array(self.board)
		newBoard[action]=self.playerTurn

		newState=GameState(newBoard, -self.playerTurn)

		value=0
		done=0

		if newState.isEndGame:
			value=newState.value[0]
			done=1

		return (newState, value, done)
	"""

    @staticmethod
    def board_to_plane_stack(board, hand1, hand2, repetitions, colour, move_count):
        state = np.zeros((len(Game.ORDER) + (2 * len(Game.HAND_ORDER)) +
                         Game.ALLOWED_REPEATS + 1 + 1, Game.BOARD_Y, Game.BOARD_X), dtype=int)
        for y in range(0, len(board)):
            for x in range(0, len(board[0])):
                if(board[y][x] != ''):
                    state[Game.ORDER.index(board[y][x])][y][x] = 1
        for i in range(0, len(hand1)):
            state[len(Game.ORDER) +
                      Game.HAND_ORDER.index(hand1[i].upper)][0][0] += 1
        for i in range(0, len(hand2)):
            state[len(Game.ORDER) + len(Game.HAND_ORDER) +
                      Game.HAND_ORDER.index(hand2[i].upper)][0][0] += 1
        for i in range(0, repetitions):
            state[len(Game.ORDER) + (2 * len(Game.HAND_ORDER)) + i][0][0] = 1
        state[len(Game.ORDER) + (2 * len(Game.HAND_ORDER)) +
                  Game.ALLOWED_REPEATS][0][0] = (1 if colour == 'W' else -1)
        state[len(Game.ORDER) + (2 * len(Game.HAND_ORDER)) +
                  Game.ALLOWED_REPEATS + 1][0][0] = move_count
        return state

    """
    convert a stack of planes to a human-friendly representation
    multiplies each layer by a number, then adds them up
    """
    @staticmethod
    def plane_stack_to_board(state):
        board = [['.' for i in range(Game.BOARD_Y)]
                                     for j in range(Game.BOARD_X)]
        state_copy = np.copy(state)
        for i in range(0, len(Game.ORDER)):
            print(state_copy[i].shape)
            state_copy[i] = np.multiply(state_copy[i], i+1)
        state_copy = np.sum(state_copy[0:len(Game.ORDER)], axis=0)
        for y in range(0, state_copy.shape[0]):
            for x in range(0, state_copy.shape[1]):
                if state_copy[y][x] != 0:
                    board[y][x] = Game.ORDER[state_copy[y][x]-1]
        hand1 = []
        for i in range(len(Game.ORDER), len(Game.ORDER) + len(Game.HAND_ORDER)):
            for _ in range(0, state[i][0][0]):
                hand1.append(Game.HAND_ORDER[i - len(Game.ORDER)])

        hand2 = []
        
        for i in range(len(Game.ORDER) + len(Game.HAND_ORDER), len(Game.ORDER) + (2 * len(Game.HAND_ORDER))):
            for _ in range(0, state[i][0][0]):
                hand2.append(Game.HAND_ORDER[i - len(Game.ORDER)].lower)

        repetitions = 0


        for i in range(len(Game.ORDER) + (2 * len(Game.HAND_ORDER)), len(Game.ORDER) + (2 * len(Game.HAND_ORDER)) + Game.ALLOWED_REPEATS):
            if state[i][0][0] == 1:
                repetitions+=1
        
        colour =  'W' if state[len(Game.ORDER) + (2 *  len(Game.HAND_ORDER)) + Game.ALLOWED_REPEATS ][0][0] == 1 else 'B'
        move_count = state[len(Game.ORDER) + (2 * len(Game.HAND_ORDER)) + Game.ALLOWED_REPEATS + 1][0][0]

        return (board, hand1, hand2, repetitions, colour, move_count)

np.set_printoptions(threshold=np.nan)
g = Game()
f = open('board.txt', 'w')
res = GameState.plane_stack_to_board(g.gameState.state)[0]

for i in range(0, len(res)):
    f.write('\t'.join(res[i]) + '\n')
