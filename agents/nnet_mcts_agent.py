import logging
import os
import random
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from itertools import groupby

import numpy as np

import config
from agents import Agent
from games import MiniShogiGame, MiniShogiGameState
from mcts import NNetMCTS
from nnets import MiniShogiNNetWrapper
from utils.normalise import normalise_examples


class NNetMCTSAgent(Agent):
    def __init__(self, nnet, comp=False, verb=False):
        super().__init__()
        self.comp = comp
        self.args = config.args
        self.skip_first_self_play = False
        self.nnet = nnet
        self.verb = verb
        self.MCTS = NNetMCTS(nnet=self.nnet)
        self.example_history = []

    def act(self, game, tau=1):
        if self.comp:
            tau = 0
        for i in range(self.args.mcts_iterations):
            #logging.debug('Playout: #{0}'.format(i))
            self.MCTS.search(game)

        action_pool = self.MCTS.action_arrays[game.game_state]
        pi_list = self.MCTS.get_action_probs(game.game_state, tau=tau)  #list

        pi_matrix = MiniShogiGameState.prob_list_to_matrix(pi_list, action_pool)

        action_tuple = (MiniShogiGameState.state_to_plane_stack(game.game_state), pi_matrix, None,
                        game.game_state.colour)

        next_action = action_pool[np.random.choice(len(action_pool), p=pi_list)]

        if self.verb:
            z, y, x = next_action
            if z < MiniShogiGame.QUEEN_ACTIONS:
                direction = z // MiniShogiGame.MAX_MOVE_MAGNITUDE
                magnitude = (z % MiniShogiGame.MAX_MOVE_MAGNITUDE) + 1
                new_x, new_y = MiniShogiGame.get_coordinates(x, y, magnitude, direction)
                x = chr(ord('a') + 5 - x - 1)
                y = y + 1
                new_x = chr(ord('a') + 5- new_x - 1)
                new_y = new_y + 1
                print('move {0}{1}{2}{3}'.format(x, y, new_x, new_y))
            elif z < MiniShogiGame.QUEEN_ACTIONS + MiniShogiGame.KNIGHT_ACTIONS:
                pass
            elif z < MiniShogiGame.QUEEN_ACTIONS + MiniShogiGame.KNIGHT_ACTIONS + MiniShogiGame.PR_QUEEN_ACTIONS:
                direction = (z - MiniShogiGame.QUEEN_ACTIONS -
                             MiniShogiGame.KNIGHT_ACTIONS) // MiniShogiGame.MAX_MOVE_MAGNITUDE
                magnitude = ((z - MiniShogiGame.QUEEN_ACTIONS - MiniShogiGame.KNIGHT_ACTIONS) %
                             MiniShogiGame.MAX_MOVE_MAGNITUDE) + 1
                new_x, new_y = MiniShogiGame.get_coordinates(x, y, magnitude, direction)
                x = chr(ord('a') + 5-x - 1)
                y = y + 1
                new_x = chr(ord('a') + 5 - new_x - 1)
                new_y = new_y + 1
                print('move {0}{1}{2}{3}!'.format(x, y, new_x, new_y))

            elif z < MiniShogiGame.QUEEN_ACTIONS + MiniShogiGame.KNIGHT_ACTIONS + MiniShogiGame.PR_QUEEN_ACTIONS + MiniShogiGame.PR_KNIGHT_ACTIONS:
                pass  # TODO PROMOTED KINGHT MOVES
            else:
                piece_index = z - MiniShogiGame.QUEEN_ACTIONS - MiniShogiGame.KNIGHT_ACTIONS - MiniShogiGame.PR_KNIGHT_ACTIONS - MiniShogiGame.PR_QUEEN_ACTIONS
                piece = MiniShogiGame.HAND_ORDER[piece_index]
                piece = piece.lower()
                x = chr(ord('a') + 5 - x - 1)
                y = y + 1
                print('move {0}@{1}{2}'.format(piece, x, y))

        #logging.debug('Value: {0}\nProbs: {1}'.format(self.MCTS.q_sa[game.game_state, next_action], self.MCTS.get_action_probs(game.game_state, tau=1)))
        #logging.debug(action_pool)
        #for action in action_pool:
        #    logging.debug(self.MCTS.n_sa[game.game_state,action])


        game.take_action(next_action)

        return action_tuple

    def run_single_game(self):
        iteration_examples = []
        game = MiniShogiGame()

        while True:
            #logging.debug('\tStep: #{0}'.format(game.game_state.move_count))
            # logging.info(game.game_state.print_state(0, flip=game.game_state.colour == 'B'))
            tau = int(game.game_state.move_count < self.args.tau)
            action = self.act(game, tau)
            iteration_examples.append(action)
            if game.game_state.game_ended():
                break
            if game.game_state.move_count > self.args.move_count_limit:  # stop very long games
                logging.warning('Game too long, terminating')
                return []

        return [(e[0], e[1], (-1) if (game.game_state.colour == e[3]) else 1) for e in iteration_examples]

    def train_neural_net(self):
        logging.info('Training {0}'.format(self.nnet.nnet.__class__.__name__))
        #self.nnet.save_checkpoint(filename='nnet0.h5')
        logging.info('Loading previous examples and model')
        try:
            self.load_examples(filename='examples_last.data')
        except Exception as e:
            logging.error(e)
        try:
            self.nnet.load_checkpoint(filename='best.h5')
        except Exception as e:
            logging.error(e)

        start_val = self.args.start_val #IMPORTANT

        for i in range(start_val, self.args.num_train_cycles+start_val):
            if (not self.skip_first_self_play) or i > start_val:
                logging.info('Generating examples')
                iteration_examples = deque([], maxlen=self.args.max_examples_len)
                # collect examples from this game

                for e in range(self.args.example_games_per_cycle):
                    if(e % (self.args.example_games_per_cycle/10) == 0):
                        logging.info('Example {0}/{1}'.format(e+1, self.args.example_games_per_cycle))
                    self.MCTS = NNetMCTS(nnet=self.nnet)
                    # collect examples from this game
                    iteration_examples += self.run_single_game()

                self.example_history.append(iteration_examples)

                if len(self.example_history) > self.args.max_example_history_len:
                    self.example_history.pop(0)

                self.save_examples(i)
            #exit(0)
            #train_examples = []
            #for e in self.example_history:
            #    train_examples.extend(e)
            train_examples = normalise_examples(self.example_history)
            random.shuffle(train_examples)
            logging.info('Starting training...')

            self.nnet.save_checkpoint(filename='temp.h5')
            new_nnet = MiniShogiNNetWrapper()
            new_nnet.load_checkpoint(filename='temp.h5')
            history = new_nnet.train(train_examples)
            new_nnet.save_checkpoint(filename='nnet{0}.h5'.format(i))
            # compare new net with previous net
            logging.info('Comparing two NNets')
            wins, nwins = self.simulate(new_nnet)

            if wins + nwins == 0 or nwins < self.args.threshold:
                logging.info('Rejecting new NN')
                self.nnet.save_checkpoint(filename='best.h5')
                #self.nnet.load_checkpoint(filename='temp.h5')
            else:
                logging.info('Accepting new NN')
                #new_nnet.save_checkpoint(filename='temp.h5')
                new_nnet.save_checkpoint(filename='best.h5')
                with open("logs/loss.log", "a") as loss_history:
                    loss_history.write(new_nnet.name + '\n')
                    loss_history.write(str(history.history) + '\n')
                self.MCTS.nnet = new_nnet
                self.nnet = new_nnet

    def simulate(self, new_nnet):
        non_draw_rounds = 0
        new_agent_wins = 0
        old_agent_wins = 0

        for i in range(1, self.args.compare_rounds + 1):
            g = MiniShogiGame()
            x = random.randint(0, 1)
            agent_w = self if x == 0 else NNetMCTSAgent(new_nnet)
            agent_b = self if x == 1 else NNetMCTSAgent(new_nnet)
            begin = time.time()
            while True:
                current_agent = agent_w if g.game_state.colour == 'W' else agent_b
                current_agent.act(g, tau=0)
                # logging.debug(g.game_state.print_state(flip=g.game_state.colour == 'B'))
                if g.game_state.game_ended():
                    if(g.game_state.colour == 'W' and x == 0) or (g.game_state.colour == 'B' and x == 1):
                        new_agent_wins += 1
                    if (g.game_state.colour == 'W' and x == 1) or (g.game_state.colour == 'B' and x == 0):
                        old_agent_wins += 1
                    non_draw_rounds += 1
                    logging.info('Stats: {0} win {1} ({2}%), {3} win {4} ({5}%)| time: {6}'.format(
                        'old_nnet', old_agent_wins, float(old_agent_wins) / non_draw_rounds * 100, 'new_nnet',
                        new_agent_wins, float(new_agent_wins) / non_draw_rounds * 100, time.time() - begin))
                    break

                if g.game_state.move_count > self.args.move_count_limit:
                    logging.warning('Game too long, terminating')
                    break

        return float(old_agent_wins) / non_draw_rounds, float(new_agent_wins) / non_draw_rounds

    def save_examples(self, iteration, folder='checkpoints', filename='examples'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename_iter = os.path.join(folder, filename + str(iteration) + '.data')
        with open(filename_iter, 'wb+') as f:
            Pickler(f).dump(self.example_history)
        filename_last = os.path.join(folder, filename + '_last' + '.data')
        with open(filename_last, 'wb+') as f:
            Pickler(f).dump(self.example_history)

    def load_examples(self, folder='checkpoints', filename='examples_last.data'):
        examples_file = os.path.join(folder, filename)

        if os.path.isfile(examples_file):
            with open(examples_file, 'rb') as f:
                self.example_history = Unpickler(f).load()
                logging.info('Examples loaded')
                x = os.path.join(folder, 'examples' + str(self.args.start_val) + '.data')
                if os.path.isfile(x):
                    logging.info(str(x) + ' loaded, skipping first self-play')
                    self.skip_first_self_play = True
                else:
                    logging.info(str(x) + ' not found, starting self-play')
        else:
            logging.warning('Example file not found')
