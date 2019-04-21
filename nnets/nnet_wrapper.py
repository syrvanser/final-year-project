from datetime import datetime
import logging
import os
import time

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
import keras.backend as K

import config
from games import MiniShogiGame, MiniShogiGameState
from nnets import MiniShogiNNetKeras, MiniShogiNNetBottleNeck, MiniShogiNNetConvResNet

class MiniShogiNNetWrapper:

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                self.nnet = MiniShogiNNetConvResNet()
                self.name = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + str(self.nnet.__class__.__name__))

                # sess = K.get_session()
                # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

                # K.set_session(sess)
                # self.session = ses

                self.nnet.model._make_predict_function()  # does not work otherwise @see https://github.com/keras-team/keras/issues/2397#issuecomment-385317242
                self.args = config.args
                self.tensorboard = TensorBoard(log_dir='logs/{0}'.format(self.name),
                                               histogram_freq=0,
                                               write_graph=True, write_images=False, write_grads=False)

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        with self.graph.as_default():
            with self.session.as_default():
                input_states, target_pis, target_vs = list(zip(*examples))
                input_states = np.asarray(input_states)

                input_states = np.swapaxes(input_states, 1, -1)

                target_pis = np.asarray(target_pis)
                target_pis = np.reshape(target_pis, (-1, MiniShogiGame.ACTION_STACK_HEIGHT * MiniShogiGame.BOARD_Y *
                                                     MiniShogiGame.BOARD_X))
                target_vs = np.asarray(target_vs)

                assert not np.any(np.isnan(input_states))
                assert not np.any(np.isnan(target_pis))
                assert not np.any(np.isnan(target_vs))
                return self.nnet.model.fit(x=input_states, y=[target_pis, target_vs],
                                              batch_size=self.args.batch_size,
                                              epochs=self.args.epochs, callbacks=[self.tensorboard], shuffle=True,
                                              validation_split=0.1)


    def predict(self, state):
        # start = time.time()
        with self.graph.as_default():
            with self.session.as_default():
                stack = MiniShogiGameState.state_to_plane_stack(state)

                stack = np.swapaxes(stack, 0, -1)

                stack = stack[np.newaxis, :, :, :]
                pi, v = self.nnet.model.predict(stack)

                pi = np.reshape(pi,
                                (-1, MiniShogiGame.ACTION_STACK_HEIGHT, MiniShogiGame.BOARD_Y, MiniShogiGame.BOARD_X))


        # logging.debug('Prediction time : {0:03f}'.format(time.time() - start))

        return pi[0], v[0][0]

    def save_checkpoint(self, folder='checkpoints', filename='weight_checkpoint.h5'):
        with self.graph.as_default():
            with self.session.as_default():
                filepath = os.path.join(folder, filename)
                if not os.path.exists(folder):
                    os.mkdir(folder)
                self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoints', filename='weight_checkpoint.h5'):
        with self.graph.as_default():
            with self.session.as_default():
                filepath = os.path.join(folder, filename)
                if not os.path.exists(filepath):
                    logging.warning("No model found in {0}".format(filepath))
                else:
                    self.nnet.model.load_weights(filepath)
