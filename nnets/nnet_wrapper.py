import logging
import os
import time
from multiprocessing import Lock
from multiprocessing.managers import BaseManager

import numpy as np

import config
from games import MiniShogiGame, MiniShogiGameState
from nnets import MiniShogiNNet
from keras import backend as K


logger = logging.getLogger(__name__)


class MiniShogiNNetWrapper:
    def __init__(self):
        self.nnet = MiniShogiNNet()
        self.mutex = Lock()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_states, target_pis, target_vs = list(zip(*examples))
        input_states = np.asarray(input_states)

        input_states = np.swapaxes(input_states, 1, -1)

        target_pis = np.asarray(target_pis)
        target_pis = np.reshape(target_pis, (-1, MiniShogiGame.ACTION_STACK_HEIGHT * MiniShogiGame.BOARD_Y *
                                             MiniShogiGame.BOARD_X))
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x=input_states, y=[target_pis, target_vs], batch_size=config.args.batch_size,
                            epochs=config.args.epochs)

    def predict(self, state):
        # start = time.time()

        stack = MiniShogiGameState.state_to_plane_stack(state)

        stack = np.swapaxes(stack, 0, -1)

        stack = stack[np.newaxis, :, :, :]

        with self.mutex:
            pi, v = self.nnet.model.predict(stack)
            K.clear_session()

        pi = np.reshape(pi, (-1, MiniShogiGame.ACTION_STACK_HEIGHT, MiniShogiGame.BOARD_Y, MiniShogiGame.BOARD_X))

        # logger.debug('Prediction time : {0:03f}'.format(time.time() - start))
        return pi[0], v[0][0]

    def save_checkpoint(self, folder='checkpoints', filename='weight_checkpoint.data'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoints', filename='weight_checkpoint.data'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model found in {0}".format(filepath))
        self.nnet.model.load_weights(filepath)


class KerasManager(BaseManager):
    pass


KerasManager.register('MiniShogiNNetWrapper', MiniShogiNNetWrapper)
