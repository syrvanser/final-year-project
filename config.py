from utils import DotDict

args = DotDict({
    'start_val': 22,  # IMPORTANT
    'lr': 0.001,  # change to 0.001?
    'dir_epsilon': 0.25,
    'dir_alpha': 0.4,  # 10 / 25
    'dropout': 0.3,  # not used
    'epochs': 5,
    'batch_size': 128,
    'reg': 0.01,
    'num_filters': 256,
    'kernel_size': 3,
    'mcts_iterations': 1000,  # num of mcts sims
    'max_depth': 300,
    'example_games_per_cycle': 200,
    'num_train_cycles': 10000,
    'c_puct': 1,
    'max_examples_len': 50000,  # train examples
    'threshold': 0.55,
    'max_example_history_len': 6,  # global examples
    'move_count_limit': 300,
    'compare_rounds': 30,
    'res_layer_num': 6,
    'tau': 15,  # was 15
})
