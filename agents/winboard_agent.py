import re

from agents import Agent
from games import MiniShogiGame


class WinBoardAgent(Agent):
    def __init__(self):
        super().__init__()

    def act(self, game):
        pass

    def winboard_to_move(self, game, wb_string):
        actions = game.game_state.allowed_actions_matrix()
        drop_regex = re.compile(r'[G,P,B,R,S]@[a-e][1-5]')
        move_regex = re.compile(r'[a-e][1-5][a-e][1-5]')

        if re.match(drop_regex, wb_string):
            piece, _, x, y = tuple(wb_string)
            if game.game_state.colour == 'W':
                x = int(ord(x) - ord('a'))
                y = 5 - int(y)
            else:
                x = int(ord('a') - ord(x) + 5 - 1)
                y = int(y) - 1
            # print('{0}, {1}'.format(x, y))
            # x -= 1
            # y = MiniShogiGame.BOARD_Y - y
            z = MiniShogiGame.QUEEN_ACTIONS + MiniShogiGame.KNIGHT_ACTIONS + MiniShogiGame.PR_QUEEN_ACTIONS + \
                MiniShogiGame.PR_KNIGHT_ACTIONS + MiniShogiGame.HAND_ORDER.index(piece)
            if actions[z][y][x] == 1:
                game.take_action((z, y, x))
            else:
                print('Error - illegal action!')
        elif re.match(move_regex, wb_string):
            if wb_string.endswith('!') or wb_string.endswith('+'):
                pr = True
            else:
                pr = False
            x, y, new_x, new_y, *_ = tuple(wb_string)

            if game.game_state.colour == 'W':
                x = int(ord(x) - ord('a'))
                y = 5 - int(y)
                new_x = int(ord(new_x) - ord('a'))
                new_y = 5 - int(new_y)
            else:
                x = int(ord('a') - ord(x) + 5 - 1)
                y = int(y) - 1
                new_x = int(ord('a') - ord(new_x) + 5 - 1)
                new_y = int(new_y) - 1

            # print('{0}, {1} -> {2}, {3}'.format(x, y, new_x,new_y))

            # x -=1
            # new_x -=1
            # y = MiniShogiGame.BOARD_Y - y
            # new_y = MiniShogiGame.BOARD_Y - new_y
            magnitude, direction = MiniShogiGame.get_direction(x, y, new_x, new_y)
            # print(magnitude)
            # print(direction)
            zd = direction * MiniShogiGame.MAX_MOVE_MAGNITUDE + (magnitude - 1)
            zp = MiniShogiGame.QUEEN_ACTIONS + MiniShogiGame.KNIGHT_ACTIONS + \
                 direction * MiniShogiGame.MAX_MOVE_MAGNITUDE + (magnitude - 1)
            # print('trying (z, y, x): (' + str(zd) + ', ' + str(y) + ', ' + str(x) + ')')
            if actions[zd][y][x] == 1 and actions[zp][y][x] == 1:
                if pr:
                    game.take_action((zp, y, x))
                else:
                    game.take_action((zd, y, x))
            elif actions[zd][y][x] == 1:
                game.take_action((zd, y, x))
            elif actions[zp][y][x] == 1:
                game.take_action((zp, y, x))
            else:
                print('Error - illegal action!')
        else:
            print('Error, {0} not recognised!'.format(wb_string))
