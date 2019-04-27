from keras.layers import *
from keras.models import *
from keras.optimizers import *

import config
from games import MiniShogiGame


class MiniShogiNNetBottleNeck2:
    def __init__(self):
        args = config.args

        # Neural Net
        input_shape = (MiniShogiGame.BOARD_Y, MiniShogiGame.BOARD_X,
                       MiniShogiGame.STATE_STACK_HEIGHT)  # s: batch_size x board_x x board_y x moves_count
        action_matrix_shape = MiniShogiGame.ACTION_STACK_HEIGHT * MiniShogiGame.BOARD_Y * MiniShogiGame.BOARD_X
        # x = Reshape()(self.input_boards)                # batch_size  x board_x x board_y x z
        nn_input = Input(shape=input_shape)
        # x_input = Reshape((Game.BOARD_Y, Game.BOARD_X, Game.STATE_STACK_HEIGHT, 1))(self.input)

        x = Conv2D(filters=32, kernel_size=3, padding='same',
                   data_format='channels_last', use_bias=False,
                   name='input_conv')(nn_input)
        x = BatchNormalization(axis=3, name='input_batch_norm')(x)
        x = ReLU(name='input_relu')(x)

        x = Conv2D(filters=64, kernel_size=3, padding='same',
                   data_format='channels_last', use_bias=False,
                   name='conv_1')(x)
        x = BatchNormalization(axis=3, name='batch_norm_1')(x)
        x = ReLU(name='relu_1')(x)

        x = Conv2D(filters=128, kernel_size=3, padding='same',
                   data_format='channels_last', use_bias=False,
                   name='conv_2')(x)
        x = BatchNormalization(axis=3, name='batch_norm_2')(x)
        x = ReLU(name='relu_2')(x)

        x = Conv2D(filters=256, kernel_size=3, padding='same',
                   data_format='channels_last', use_bias=False,
                   name='conv_3')(x)
        x = BatchNormalization(axis=3, name='batch_norm_3')(x)
        x = ReLU(name='relu_3')(x)

        v = Conv2D(filters=1, kernel_size=1, padding='same',
                   data_format='channels_last', use_bias=False,
                   name='value_conv')(x)

        v = BatchNormalization(axis=3, name='value_batch_norm')(v)
        v = ReLU(name='value_relu')(v)

        v = Flatten(name='value_flatten')(v)
        v = Dropout(args.dropout, name='value_dropout')(v)

        v = Dense(1, activation='tanh', name='value_dense_out', use_bias=False)(v)

        pi = Conv2D(filters=84, kernel_size=1, padding='same', data_format='channels_last', use_bias=False,
                    name='policy_conv_2')(x)
        pi = BatchNormalization(axis=3, name='policy_batch_norm_2')(pi)
        pi = ReLU(name='policy_relu_2')(pi)
        pi = Flatten(name='policy_flatten')(pi)
        pi = Dropout(args.dropout, name='policy_dropout')(pi)
        pi = Dense(action_matrix_shape, activation='softmax', name='policy_dense_out', use_bias=False)(pi)

        self.model = Model(inputs=nn_input, outputs=[pi, v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
