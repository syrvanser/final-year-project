import logging
import os
import random
import time
from collections import deque
from pickle import Pickler, Unpickler

import numpy as np
import multiprocessing as mp

import config
from agents import Agent
from games import MiniShogiGame, MiniShogiGameState
from mcts import NNetMCTS
from nnets import MiniShogiNNetWrapper, KerasManager

logger = logging.getLogger(__name__)


class NNetMCTSAgent(Agent):
    def __init__(self, nnet):

        super().__init__()
        self.skip_first_self_play = False
        self.nnet = nnet
        self.MCTS = NNetMCTS(nnet=self.nnet)
        self.example_history = []

    def act(self, game):
        self.parallel_safe_act(game, self.MCTS)

    @staticmethod
    def parallel_safe_act(game, mcts):
        for i in range(config.args.mcts_iterations):
            #logger.debug('Playout: #{0}'.format(i))
            mcts.search(game)

        action_tuple = (MiniShogiGameState.state_to_plane_stack(game.game_state), mcts.p_s[game.game_state], None,
                        game.game_state.colour)

        action_pool = mcts.action_arrays[game.game_state]
        next_action = action_pool[np.random.choice(len(action_pool), p=mcts.get_action_probs(game.game_state))]

        game.take_action(next_action)

        return action_tuple

    @classmethod
    def run_single_game(cls, nnet):
        iteration_examples = []
        game = MiniShogiGame()
        mcts = NNetMCTS(nnet=nnet)

        while True:  # set max game duration?
            # logger.debug('\tStep: #{0}'.format(game.game_state.move_count))
            # logger.info(game.game_state.print_state(0, flip=game.game_state.colour == 'B'))
            action = cls.parallel_safe_act(game, mcts)
            iteration_examples.append(action)
            if game.game_state.game_ended():
                break
            if game.game_state.move_count > config.args.move_count_limit:  # just stop very long games
                #logger.warning('Game too long, terminating')
                return []

        return [(e[0], e[1], (-1) ** (game.game_state.colour == e[3])) for e in iteration_examples]

    def train_neural_net(self):

        logger.info('Loading previous examples and model')
        try:
            self.nnet.load_checkpoint(filename='temp.data')
            self.load_examples(filename='examples{0}.data'.format(config.args.example_iter_number))
        except Exception as e:
            logger.error(e)

        for i in range(config.args.num_epochs):
            if (not self.skip_first_self_play) or i > 0:
                logger.info('Generating examples')
                iteration_examples = deque([], maxlen=config.args.max_examples_len)
                # self.MCTS = NNetMCTS(nnet=self.nnet, args=self.args)
                # collect examples from this game

                '''with mp.Pool(processes=config.args.processes) as pool:
                    # iteration_examples += self.run_single_game()
                    with KerasManager() as manager:
                        nnet = manager.MiniShogiNNetWrapper()
                        results = [pool.apply_async(self.run_single_game, (nnet,)) for _ in
                                   range(0, config.args.max_example_games)]
                        output = [p.get() for p in results]
                        for o in output:
                            iteration_examples += o
                '''

                with KerasManager() as manager:
                    nnet = manager.MiniShogiNNetWrapper()
                    result = self.run_single_game(nnet)
                    print('123')
                    exit(0)
                    #for o in output:
                    #    iteration_examples += o

                self.example_history.append(iteration_examples)

                if len(self.example_history) > config.args.max_example_history_len:
                    self.example_history.pop(0)

                self.save_examples(i)

            train_examples = []
            for e in self.example_history:
                train_examples.extend(e)
            random.shuffle(train_examples)

            logger.info('Starting training...')

            self.nnet.save_checkpoint(filename='temp.data')
            new_nnet = MiniShogiNNetWrapper()
            new_nnet.load_checkpoint(filename='temp.data')
            new_nnet.train(train_examples)

            # compare new net with previous net
            logger.info('Comparing two NNets')
            wins, nwins = self.simulate(new_nnet)

            if wins + nwins == 0 or float(nwins) / (wins + nwins) < config.args.threshold:
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
        non_draw_rounds = 0
        new_agent_wins = 0
        old_agent_wins = 0

        for i in range(1, config.args.compare_rounds + 1):
            g = MiniShogiGame()
            x = random.randint(0, 1)
            agent1 = self if x == 0 else NNetMCTSAgent(new_nnet)
            agent2 = self if x == 1 else NNetMCTSAgent(new_nnet)
            begin = time.time()
            while True:
                current_agent = agent1 if g.game_state.colour == 'W' else agent2
                current_agent.act(g)
                # logger.debug(g.game_state.print_state(flip=g.game_state.colour == 'B'))
                if g.game_state.game_ended():
                    if(g.game_state.colour == 'W' and x == 0) or (g.game_state.colour == 'B' and x == 1):
                        old_agent_wins += 1
                    if (g.game_state.colour == 'W' and x == 1) or (g.game_state.colour == 'B' and x == 0):
                        new_agent_wins += 1
                    non_draw_rounds += 1
                    logger.info('Stats: {0} win {1} ({2}%), {3} win {4} ({5}%)| time: {6}'.format(
                        'old_nnet', old_agent_wins, old_agent_wins / non_draw_rounds * 100, 'new_nnet',
                        new_agent_wins, new_agent_wins / non_draw_rounds * 100, time.time() - begin))
                    break

                if g.game_state.move_count > config.args.move_count_limit:
                    logger.warning('Game too long, terminating')

        return old_agent_wins / non_draw_rounds, new_agent_wins / non_draw_rounds

    def save_examples(self, iteration, folder='checkpoints', filename='examples'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, filename + str(iteration) + '.data')
        with open(filename, 'wb+') as f:
            Pickler(f).dump(self.example_history)

    def load_examples(self, folder='checkpoints', filename='examples0.data'):
        examples_file = os.path.join(folder, filename)

        if os.path.isfile(examples_file):
            with open(examples_file, 'rb') as f:
                self.example_history = Unpickler(f).load()
                self.skip_first_self_play = True
                logger.info('Examples loaded, skipping first self play')
        else:
            logger.warning('Example file not found, generating...')
