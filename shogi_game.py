import numpy as np
from bidict import bidict

class Game:

    BOARD_X = 9
    BOARD_Y = 9
    ALLOWED_REPEATS = 3
    INITIAL_LAYOUT = [['l', 'n', 's', 'g', 'k', 'g', 's', 'n', 'l'],
            ['', 'r', '', '', '', '', '', 'b', ''],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['', '', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', '', ''],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['', 'B', '', '', '', '', '', 'R', ''],
            ['L', 'N', 'S', 'G', 'K', 'G', 'S', 'N', 'L']]

    """dictionary = bidict({'pawn': 'P', 'rook': 'R', 'bishop': 'B', 'lance': 'L', 'silver_general': 'S', 'golden_general': 'G', 'kight': 'N',
                         'king': 'K', 'horse': '+H', 'dragon': '+B', 'tokin': '+P', 'promoted_silver': '+S', 
                         'promoted_lance': '+L', 'promoted_knight': '+N'})"""
    order = [ 'P', 'L', 'N', 'S', 'G', 'B', 'R', '+P', '+L', '+N', '+S', '+B', '+R', 'K', 'p', 'l', 'n', 's', 'g', 'b', 'r', '+p', '+l', '+n', '+s', '+b', '+r', 'k']
    hand_order = ['P', 'L', 'N', 'S', 'G', 'B', 'R']

    def __init__(self):
            self.reset()

    def reset(self):
        self.gameState = GameState(GameState.game_to_3darray(Game.INITIAL_LAYOUT, [], [], 0, 0, 0))
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
    """
	def _allowedActions(self):
		allowed=[]

		return allowed


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
    def game_to_3darray(board, hand1, hand2, repetitions, colour, move_count):
        f = open("board.txt", "w")
        state = np.zeros((47, Game.BOARD_Y, Game.BOARD_X))
        for y in range(0, len(board)):
            for x in range(0, len(board[0])):
                if(board[y][x]!= ''):
                    state[Game.order.index(board[y][x])][y][x] = 1
        for i in range(0, len(hand1)):
            state[len(Game.order) + Game.hand_order.index(hand1[i].upper)][0][0] += 1
        for i in range(0, len(hand1)):
            state[len(Game.order) + len(Game.hand_order) + Game.hand_order.index(hand2[i].upper)][0][0] += 1
        for i in range(0, repetitions):
            state[len(Game.order) + len(Game.hand_order) + len(Game.hand_order) + i][0][0] = 1
        state[len(Game.order) + len(Game.hand_order) + len(Game.hand_order) + Game.ALLOWED_REPEATS ][0][0] = (0 if colour == 'W' else 1)
        state[len(Game.order) + len(Game.hand_order) + len(Game.hand_order) + Game.ALLOWED_REPEATS + 1][0][0] = move_count

        f.write(np.array2string(state))

np.set_printoptions(threshold=np.nan)
Game()
