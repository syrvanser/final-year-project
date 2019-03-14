from utils import DotDict

args = DotDict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 20,
    'batch_size': 64,
    'reg': 1e-4,
    'num_filters': 256,
    'kernel_size': 3,
    'mcts_iterations': 100,  # num of mcts sims
    'max_depth': 300,
    'max_example_games': 50,
    'num_epochs': 100,
    'c_puct': 1,
    'max_examples_len': 200000,  # train examples
    'threshold': 0.6,
    'max_example_history_len': 20,  # global examples
    'move_count_limit': 500,
    'compare_rounds': 40,
    'res_layer_num': 5,
    'tau': 15000 #was 15
})
