import logging
import re

from agents import Agent
from games import MiniShogiGame

logger = logging.getLogger(__name__)


class HumanShogiAgent(Agent):
    def __init__(self):
        super().__init__()

    def act(self, game):
        actions = game.game_state.allowed_actions_matrix()
        print(game.game_state.print_state())
        drop_regex = re.compile(r'[G,P,B,R,S]\s[0-4]\s[0-4]')
        move_regex = re.compile(r'[0-4]\s[0-4]\s[0-4]\s[0-4](\sD)?')
        while True:
            """action_pool = []
            for y in range(0, game.BOARD_Y):
                for x in range(0, game.BOARD_X):
                    for z in range(0, len(actions[0][0])):
                        if(actions[y][x][z] == 1):
                            action_pool.append((x, y, z))
            print(action_pool)"""
            choice = input('> ')
            if re.match(drop_regex, choice):
                piece, x, y = choice.split(' ')
                x = int(x)
                y = int(y)
                z = MiniShogiGame.QUEEN_ACTIONS + MiniShogiGame.KNIGHT_ACTIONS + MiniShogiGame.PR_QUEEN_ACTIONS + \
                    MiniShogiGame.PR_KNIGHT_ACTIONS + MiniShogiGame.HAND_ORDER.index(piece)
                if actions[z][y][x] == 1:
                    game.take_action((x, y, z))
                    break
                else:
                    print('Illegal drop')
            elif re.match(move_regex, choice):
                x, y, new_x, new_y = tuple(map(int, choice.split(' ')))
                magnitude, direction = MiniShogiGame.get_direction(x, y, new_x, new_y)
                if choice[-1] != 'P':
                    z = direction * MiniShogiGame.MAX_MOVE_MAGNITUDE + (magnitude - 1)
                else:
                    z = MiniShogiGame.QUEEN_ACTIONS + MiniShogiGame.KNIGHT_ACTIONS + \
                        direction * MiniShogiGame.MAX_MOVE_MAGNITUDE + (magnitude - 1)
                # print('trying (x, y, z): (' + str(x) + ', ' + str(y) + ', ' + str(z) + ')')
                if actions[z][y][x] == 1:
                    game.take_action((x, y, z))
                    break
                else:
                    print('Illegal move')

            else:
                print('Wrong Input')
