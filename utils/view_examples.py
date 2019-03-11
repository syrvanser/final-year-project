import logging
import os

import numpy as np
from pickle import Unpickler


def load_examples(folder='checkpoints', filename='examples0.data'):
    examples_file = os.path.join(folder, filename)
    with open(examples_file, 'rb') as f:
        return Unpickler(f).load()

logging.basicConfig(format=' %(asctime)s %(name)-30s %(levelname)-8s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p:',
                    filename='logs/examples.log',
                    filemode='w',
                    level=logging.INFO)
np.set_printoptions(threshold=np.nan)

logging.info('Logging examples0.data:')
examples = load_examples()
input_states, target_pis, target_vs = list(zip(*examples))
logging.info(len(examples))
logging.info(target_vs)
