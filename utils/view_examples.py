import logging
import os

import numpy as np
from pickle import Unpickler


def load_examples(folder='checkpoints', filename='examples0.data'):
    examples_file = os.path.join(folder, filename)
    with open(examples_file, 'rb') as f:
        return Unpickler(f).load()


logger = logging.getLogger(__name__)
logging.basicConfig(format=' %(asctime)s %(name)-30s %(levelname)-8s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p:',
                    filename='examples.log',
                    filemode='w',
                    level=logging.INFO)
np.set_printoptions(threshold=np.nan)

examples = load_examples()
input_states, target_pis, target_vs = list(zip(*examples))
logger.info(len(examples))
logger.info(target_vs)
