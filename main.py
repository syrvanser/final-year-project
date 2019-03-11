import logging
import random
import time

import numpy as np

from agents import random_agent, nnet_mcts_agent
from games import mini_shogi_game
from nnets.nnet_wrapper import MiniShogiNNetWrapper

logger = logging.getLogger(__name__)


def play():
    rounds = 1
    white_wins = 0
    agent1 = random_agent.RandomAgent()
    agent2 = nnet_mcts_agent.NNetMCTSAgent(
        MiniShogiNNetWrapper())
    #print('Preparing neural net')
    #agent2.train_neural_net()
    #print('Preparation complete')
    for i in range(1, rounds + 1):
        begin = time.time()
        print('Game {0}/{1}'.format(i, rounds))
        g = mini_shogi_game.MiniShogiGame()
        while True:
            current_agent = agent1 if g.game_state.colour == 'W' else agent2
            current_agent.act(g)
            # print(g.game_state.print())
            logger.info(g.game_state.print_state(flip=g.game_state.colour == 'B'))
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
                logger.info('Stats: {0} win {1} ({2}%), {3} win {4} ({5}%)| time: {6}'.format(
                    agent1.__class__.__name__, white_wins, white_wins /
                    i * 100, agent2.__class__.__name__,
                    i - white_wins, (i - white_wins) / i * 100,
                    time.time() - begin))
                break


if __name__ == '__main__':
    # print(Game.get_coordinates(0,3,0,2))
    random.seed(1)
    logging.basicConfig(format=' %(asctime)s %(name)-30s %(levelname)-8s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p:',
                        filename='game.log',
                        filemode='w',
                        level=logging.DEBUG)
    np.set_printoptions(threshold=np.nan)
    play()
