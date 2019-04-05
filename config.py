from utils import DotDict

args = DotDict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 40,
    'batch_size': 64,
    'reg': 0.01,
    'num_filters': 256,
    'kernel_size': 3,
    'mcts_iterations': 400,  # num of mcts sims
    'max_depth': 300,
    'example_games_per_cycle': 50,
    'num_train_cycles': 100,
    'c_puct': 1,
    'max_examples_len': 20000,  # train examples
    'threshold': 0.6,
    'max_example_history_len': 100,  # global examples
    'move_count_limit': 400,
    'compare_rounds': 40,
    'res_layer_num': 8,
    'tau': 15000, #was 15
})
