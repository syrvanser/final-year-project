import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
import logging
logger = logging.getLogger(__name__)
from nnet import MiniShogiNNet as msnn

import argparse

args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': True,
    'num_channels': 256,
    'filter_size': 5
}

class NNetWrapper:
    def __init__(self):
        self.nnet = msnn(args)

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_states, target_pis, target_vs = list(zip(*examples))
        input_states = np.asarray(input_states)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x = input_states, y = [target_pis, target_vs], batch_size = args['batch_size'], epochs = args['epochs'])

    def predict(self, state):
        start = time.time()

        pi, v = self.nnet.model.predict(state)

        logger.log('Prediction time : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoints', filename='weight_checkpoint.data'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.nnet.model.save_weights(filepath)
 
    def load_checkpoint(self, folder='checkpoints', filename='weight_checkpoint.data'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model found in {0}".format(filepath))
        self.nnet.model.load_weights(filepath)