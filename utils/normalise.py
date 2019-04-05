import os
from hashlib import sha1
from pickle import Unpickler
from sklearn.preprocessing import normalize


examples_file = os.path.join('checkpoints', 'examples_last.data')
if os.path.isfile(examples_file):
    with open(examples_file, 'rb') as f:
        example_history = Unpickler(f).load()
        print('Examples loaded')
        flat_examples = [item for sublist in example_history for item in sublist]

        print(len(flat_examples))
        aggr = dict()
        for example in flat_examples:
            state_hash = hash(str((example[0])))
            if state_hash in aggr:
                element = aggr[state_hash]
                aggr[state_hash] = (example[0], [sum(x) for x in zip(element[1], example[1])], element[2] + example[2], element[3]+1)
            else:
                aggr[state_hash] = (example[0], example[1], example[2], 1)

        normalised = []
        for state_hash in aggr:
            element = aggr[state_hash]
            norm_pi = [x / element[3] for x in element[1]]

            norm_v = element[2] / element[3]
            normalised.append((element[0], norm_pi, norm_v))

        print(len(normalised))
        print(normalised[0])



else:
    print('Examples file not found')




