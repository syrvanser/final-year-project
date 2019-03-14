import logging
import random
import sys
import time

import numpy as np

from agents import HumanAgent, NNetMCTSAgent, RandomAgent, BasicMCTSAgent
from games import mini_shogi_game
from nnets.nnet_wrapper import MiniShogiNNetWrapper
from keras.utils.vis_utils import plot_model


def play():
    rounds = 100
    white_wins = 0
    agent1 = RandomAgent()
    nnet = MiniShogiNNetWrapper()
    nnet.nnet.model.summary()
    plot_model(nnet.nnet.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    agent2 = NNetMCTSAgent(nnet)
    print('Preparing neural net')
    agent2.train_neural_net()
    agent2.nnet.load_checkpoint(filename='best.h5')

    print('Preparation complete')
    for i in range(1, rounds + 1):
        begin = time.time()
        print('Game {0}/{1}'.format(i, rounds))
        g = mini_shogi_game.MiniShogiGame()
        while True:
            current_agent = agent1 if g.game_state.colour == 'W' else agent2
            current_agent.act(g)
            # print(g.game_state.print())
            logging.info(g.game_state.print_state(flip=g.game_state.colour == 'B'))
            if g.game_state.game_ended():
                if g.game_state.colour == 'B':
                    white_wins += 1
                print('Stats: {0} win {1} ({2}%), {3} win {4} ({5}%)| time: {6}'.format(agent1.__class__.__name__,
                                                                                        white_wins,
                                                                                        white_wins / i * 100,
                                                                                        agent2.__class__.__name__,
                                                                                        i - white_wins,
                                                                                        (i - white_wins) / i * 100,
                                                                                        time.time() - begin))
                logging.info('Stats: {0} win {1} ({2}%), {3} win {4} ({5}%)| time: {6}'.format(
                    agent1.__class__.__name__, white_wins, white_wins /
                    i * 100, agent2.__class__.__name__,
                    i - white_wins, (i - white_wins) / i * 100,
                    time.time() - begin))
                break


if __name__ == '__main__':
    #random.seed(1)
    logging.basicConfig(format=' %(asctime)s %(name)-30s %(levelname)-8s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p:',
                        filename='logs/game.log',
                        filemode='w',
                        level=logging.INFO)
    np.set_printoptions(threshold=sys.maxsize)
    play()
