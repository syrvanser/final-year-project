from keras.layers import *
from keras.models import *
from keras.optimizers import *

import config
from games import MiniShogiGame


class MiniShogiNNetBottleNeck:
    def __init__(self):
        args = config.args

        # Neural Net
        input_shape = (MiniShogiGame.BOARD_Y, MiniShogiGame.BOARD_X,
                       MiniShogiGame.STATE_STACK_HEIGHT)  # s: batch_size x board_x x board_y x moves_count
        action_matrix_shape = MiniShogiGame.ACTION_STACK_HEIGHT * MiniShogiGame.BOARD_Y * MiniShogiGame.BOARD_X
        # x = Reshape()(self.input_boards)                # batch_size  x board_x x board_y x z
        nn_input = Input(shape=input_shape)
        # x_input = Reshape((Game.BOARD_Y, Game.BOARD_X, Game.STATE_STACK_HEIGHT, 1))(self.input)

        x = Conv2D(filters=args.num_filters, kernel_size=3, padding='same',
                   data_format='channels_last', use_bias=False,
                   name='input_conv')(nn_input)
        x = BatchNormalization(axis=3, name='input_batch_norm')(x)
        x = ReLU(name='input_relu')(x)

        x = Conv2D(filters=args.num_filters, kernel_size=3, padding='same',
                   data_format='channels_last', use_bias=False,
                   name='conv_1')(x)
        x = BatchNormalization(axis=3, name='batch_norm_1')(x)
        x = ReLU(name='relu_1')(x)

        x = Conv2D(filters=args.num_filters, kernel_size=3, padding='same',
                   data_format='channels_last', use_bias=False,
                   name='conv_2')(x)
        x = BatchNormalization(axis=3, name='batch_norm_2')(x)
        x = ReLU(name='relu_2')(x)

        x = Conv2D(filters=args.num_filters, kernel_size=3, padding='same',
                   data_format='channels_last', use_bias=False,
                   name='conv_3')(x)
        x = BatchNormalization(axis=3, name='batch_norm_3')(x)
        x = ReLU(name='relu_3')(x)

        x = Flatten(name='dense_flatten_1')(x)
        x = Dense(1024, use_bias=False, name='dense_1')(x)  # batch_size x 1024
        x = BatchNormalization(axis=1, name='batch_norm_dense_1')(x)
        x = Activation('relu', name='relu_dense_1')(x)

        x = Dense(512, use_bias=False, name='dense_2')(x)  # batch_size x 1024
        x = BatchNormalization(axis=1, name='batch_norm_dense_2')(x)
        x = Activation('relu', name='relu_dense_2')(x)

        pi = Dense(action_matrix_shape, activation='softmax', name='pi_out')(x)  # batch_size x self.action_size
        v = Dense(1, activation='tanh', name='v_out')(x)  # batch_size x 1

        self.model = Model(inputs=nn_input, outputs=[pi, v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
