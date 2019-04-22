import logging
import re

from agents import Agent
from games import MiniShogiGame, MiniShogiGameState
from pandas import DataFrame


class HumanAgent(Agent):
    def __init__(self):
        super().__init__()

    def act(self, game):
        actions = game.game_state.allowed_actions_matrix()
        print(DataFrame(game.game_state.board))
        print('Your hand: {0}\nOpponents hand: {1}'.format(game.game_state.hand1, game.game_state.hand2))
        drop_regex = re.compile(r'[G,P,B,R,S]\s[0-4]\s[0-4]')
        move_regex = re.compile(r'[0-4]\s[0-4]\s[0-4]\s[0-4]')
        while True:
            #action_pool = MiniShogiGameState.action_matrix_to_action_array(actions)
            #print(action_pool)
            choice = input('> ')
            if re.match(drop_regex, choice):
                piece, x, y = choice.split(' ')
                x = int(x)
                y = int(y)
                #x -= 1
                #y = MiniShogiGame.BOARD_Y - y
                z = MiniShogiGame.QUEEN_ACTIONS + MiniShogiGame.KNIGHT_ACTIONS + MiniShogiGame.PR_QUEEN_ACTIONS + \
                    MiniShogiGame.PR_KNIGHT_ACTIONS + MiniShogiGame.HAND_ORDER.index(piece)
                if actions[z][y][x] == 1:
                    game.take_action((z, y, x))
                    break
                else:
                    print('Illegal drop')
            elif re.match(move_regex, choice):
                x, y, new_x, new_y = tuple(map(int, choice.split(' ')))
                #x -=1
                #new_x -=1
                #y = MiniShogiGame.BOARD_Y - y
                #new_y = MiniShogiGame.BOARD_Y - new_y

                magnitude, direction = MiniShogiGame.get_direction(x, y, new_x, new_y)
                #print(magnitude)
                #print(direction)
                zd = direction * MiniShogiGame.MAX_MOVE_MAGNITUDE + (magnitude - 1)
                zp = MiniShogiGame.QUEEN_ACTIONS + MiniShogiGame.KNIGHT_ACTIONS + \
                        direction * MiniShogiGame.MAX_MOVE_MAGNITUDE + (magnitude - 1)
                print('trying (z, y, x): (' + str(zd) + ', ' + str(y) + ', ' + str(x) + ')')
                if actions[zd][y][x] == 1 and actions[zp][y][x] == 1:
                    print('Promote (y)/(n)?')
                    inp = input('>')
                    if inp == 'y':
                        game.take_action((zp, y, x))
                        break
                    elif inp == 'n':
                        game.take_action((zd, y, x))
                        break
                    else:
                        print('Wrong Input')
                elif actions[zd][y][x] == 1:
                    game.take_action((zd, y, x))
                    break
                elif actions[zp][y][x] == 1:
                    game.take_action((zp, y, x))
                    break
                else:
                    print('Illegal move')

            else:
                print('Wrong Input')
