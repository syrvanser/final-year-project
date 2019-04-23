import logging
import sys
import numpy as np
import os
import tensorflow as tf

from agents import NNetMCTSAgent, WinBoardAgent
from games import MiniShogiGame
from nnets import MiniShogiNNetWrapper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)
INITIAL = 'setboard rbsgk/4p/5/P4/KGSBR[-] w 0 1'


def main():
    logging.basicConfig(format=' %(asctime)s %(module)-30s %(levelname)-8s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p:',
                        filename='logs/game.log',
                        filemode='a',
                        level=logging.INFO)
    np.set_printoptions(threshold=sys.maxsize)

    stack = []
    turn = True
    forced = False
    while True:
        if stack:
            command = stack.pop()
        else:
            command = input()

        if command == 'quit':
            break

        elif command == 'protover 2':
            print('feature setboard=1 usermove=1 myname="MiniShogiEngine"')
            print('feature option="multiPV margin -spin 0 0 1000"')
            print('feature ping=1 variants="shogi,mini,judkin,5x5+5_shogi,6x6+6_shogi,7x7+6_shogi,tori,8x8+5_shogi,euroshogi" memory=1 reuse=0 done=1')
        elif command == 'new':
            stack.append('setboard')

        elif command.startswith('setboard'):
            g = MiniShogiGame()
            nnet = MiniShogiNNetWrapper()
            # nnet.nnet.model.summary()
            winboard_agent = WinBoardAgent()
            agent = NNetMCTSAgent(nnet, comp=True, verb=True)
            agent.nnet.load_checkpoint(filename='best.h5')

        elif command == 'force':
            forced = True

        elif command == 'go':
            if turn:
                agent.act(g)
                logging.debug(g.game_state.print_state())
                turn = False


        elif command.startswith('ping'):
            _, m = command.split()
            print('pong', m)

        elif command.startswith('usermove'):
            turn = True
            _, command = command.split()
            winboard_agent.winboard_to_move(g, command)
            logging.debug(g.game_state.print_state())
            if forced:
                stack.append('go')

        elif any(command.startswith(x) for x in ('xboard', 'random', 'hard', 'accepted', 'level')):
            pass

        else:
            print("Error (unkown command):", command)

if __name__ == '__main__':
    main()