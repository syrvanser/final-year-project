import logging
import random
import sys
import time

import tensorflow as tf

import numpy as np

from agents import HumanAgent, NNetMCTSAgent, RandomAgent, BasicMCTSAgent
from games import MiniShogiGame
from nnets import MiniShogiNNetWrapper
from keras.utils.vis_utils import plot_model


def play():
    rounds = 30
    white_wins = 0
    agent1 = HumanAgent()
    nnet = MiniShogiNNetWrapper()
    #nnet.nnet.model.summary()
    plot_model(nnet.nnet.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    agent2 = NNetMCTSAgent(nnet, comp=False)
    print('Preparing neural net')
    #agent2.train_neural_net()
    agent2.comp = True
    agent2.nnet.load_checkpoint(filename='best.h5')

    print('Preparation complete')
    for i in range(1, rounds + 1):
        begin = time.time()
        print('Game {0}/{1}'.format(i, rounds))
        g = MiniShogiGame()
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
            if g.game_state.move_count > 300:  # stop very long games
                print('Game too long, terminating')
                break

def play_first_gen():
    nnet1 = MiniShogiNNetWrapper()
    nnet2 = MiniShogiNNetWrapper()
    agent1 = NNetMCTSAgent(nnet1, comp=True)
    nnet1.load_checkpoint(filename='raw.h5')
    nnet2.load_checkpoint(filename='best_temp.h5')
    agent1.simulate(nnet2)

if __name__ == '__main__':
    #seed = 42
    #random.seed(seed)
    #np.random.seed(seed)
    #tf.set_random_seed(seed)
    logging.basicConfig(format=' %(asctime)s %(module)-30s %(levelname)-8s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p:',
                        filename='logs/game.log',
                        filemode='a',
                        level=logging.DEBUG)
    np.set_printoptions(threshold=sys.maxsize)
    #play_first_gen()
    #play()

    nnet = MiniShogiNNetWrapper()
    nnet.nnet.model.summary()
    plot_model(nnet.nnet.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    agent = NNetMCTSAgent(nnet, comp=False)
    print('Training neural net')
    agent.train_neural_net()
