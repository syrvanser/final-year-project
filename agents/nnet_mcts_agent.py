import logging
import os
import random
import time
from collections import deque
from copy import deepcopy
from pickle import Pickler, Unpickler

import numpy as np

from agents.agent import Agent
from games.mini_shogi_game import MiniShogiGame, MiniShogiGameState
from mcts.nnet_mcts import NNetMCTS

logger = logging.getLogger(__name__)


class NNetMCTSAgent(Agent):
    def __init__(self, nnet, args):

        super().__init__()
        self.args = args

        self.nnet = nnet
        self.MCTS = NNetMCTS(nnet=self.nnet, args=self.args)
        self.example_history = []
        self.iteration_examples = []

    def act(self, game):

        for i in range(self.args.limit):
            logger.debug('Playout: #{0}'.format(i))
            self.MCTS.search(game)

        action_pool = self.MCTS.action_arrays[game.game_state]
        next_action = action_pool[np.random.choice(len(action_pool), p=self.MCTS.get_action_probs(game.game_state))]

        game.take_action(next_action)

    def run_single_game(self):
        self.iteration_examples = []
        game = MiniShogiGame()
        while True:     # set max game duration?
            logger.info('\tStep: #{0}'.format(game.game_state.move_count))
            logger.info(game.game_state.print_state(0, flip=game.game_state.colour == 'B'))
            self.act(game)
            if game.game_state.game_ended():
                break

        for example in self.iteration_examples:
            if game.game_state.colour == example[0].colour:
                example[2] = -1
            else:
                example[2] = 1
        return self.iteration_examples

    def train_neural_net(self):
        
        logger.info('Loading previous examples and model')
        try:
            self.nnet.load_checkpoint(filename='temp.data')
            self.example_history = self.load_examples(self.args.example_iter_number)
        except Exception as e:
            logger.error(e)
        
        for i in range(self.args.num_epochs):
            iteration_examples = deque([], maxlen=self.args.max_examples_len)
            logger.info('Generating examples')
            for e in range(self.args.max_example_games):
                logger.info('Example {0}/{1}'.format(e, self.args.max_example_games))
                self.MCTS = NNetMCTS(nnet=self.nnet, args=self.args)
                # collect examples from this game
                iteration_examples += self.run_single_game()
            self.example_history.append(iteration_examples)

            if len(self.example_history) > self.args.max_example_history_len:
                self.example_history.pop(0)

            self.save_examples(iteration_examples, i)

            random.shuffle(iteration_examples)

            logger.info('Example generation finished, starting training...')

            self.nnet.save_checkpoint(filename='temp.data')
            new_nnet = deepcopy(self.nnet)
            new_nnet.load_checkpoint(filename='temp.data')
            new_nnet.train(iteration_examples)

            # compare new net with previous net
            wins, nwins = self.simulate(new_nnet)
            logger.info('Comparing two NNets')
            if wins + nwins == 0 or float(nwins) / (wins + nwins) < self.args.threshold:
                logger.info('Rejecting new NN')
                self.nnet.load_checkpoint(filename='temp.data')
            else:
                logger.info('Accepting new NN')
                self.nnet.save_checkpoint(filename='temp.data')
                self.nnet.save_checkpoint(filename='best.data')
                nnet = new_nnet  # replace with new net
                self.MCTS.nnet = nnet
                self.nnet = nnet

    def simulate(self, new_nnet):
        rounds = 100
        white_wins = 0
        for i in range(1, rounds + 1):
            g = MiniShogiGame()
            x = random.randint(0, 1)
            agent1 = self if x == 0 else NNetMCTSAgent(new_nnet, self.args)
            agent2 = self if x == 1 else NNetMCTSAgent(new_nnet, self.args)
            begin = time.time()
            while True:
                current_agent = agent1 if g.game_state.colour == 'W' else agent2
                current_agent.act(g)
                # print(g.game_state.print())
                logger.debug(g.game_state.print_state(
                    flip=g.game_state.colour == 'B'))
                if g.game_state.game_ended():
                    if g.game_state.colour == 'B':
                        white_wins += 1
                    logger.info('Stats: {0} win {1} ({2}%), {3} win {4} ({5}%)| time: {6}'.format(
                        agent1.__class__.__name__, white_wins, white_wins / i * 100, agent2.__class__.__name__,
                        i - white_wins, (i - white_wins) / i * 100, time.time() - begin))
                    break

            if x == 0:
                return white_wins / rounds, (rounds - white_wins) / rounds
            else:
                return (rounds - white_wins) / rounds, white_wins / rounds

    @staticmethod
    def save_examples(train_examples, iteration, folder='checkpoints', filename='examples'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, filename + str(iteration) + '.data')
        with open(filename, 'wb+') as f:
            Pickler(f).dump(train_examples)

    @staticmethod
    def load_examples(folder='checkpoints', filename='examples.data'):
        examples_file = os.path.join(folder, filename)
        with open(examples_file, 'rb') as f:
            return Unpickler(f).load()
