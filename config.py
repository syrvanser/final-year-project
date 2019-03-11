from utils import DotDict

args = DotDict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 512,
    'filter_size': 3,
    'mcts_iterations': 50,  # num of mcts sims
    'max_depth': 200,
    'max_example_games': 10,
    'num_epochs': 100,
    'c_puct': 1,
    'max_examples_len': 100000,  # train examples
    'threshold': 0.6,
    'max_example_history_len': 20,  # global examples
    'example_iter_number': 2,
    'move_count_limit': 500,
    'compare_rounds': 10
})