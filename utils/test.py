import os
from pickle import Unpickler, Pickler

from utils.normalise import normalise_examples


def test():
    examples_file = os.path.join('checkpoints', 'examples_last.data')
    if os.path.isfile(examples_file):
        with open(examples_file, 'rb') as f:
            example_history = Unpickler(f).load()
            print('Examples loaded')
            for list in example_history:
                for element in list:
                    if element[2] == 0.4117559416775118:
                        print('found!')
                        exit(0)
                    else:
                        print(element[2])
            # with open('checkpoints/examples_normalised.data', 'wb+') as f:
            #    Pickler(f).dump(example_history)
    else:
        print('examples file not found')


test()
