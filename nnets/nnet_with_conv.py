from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2

import config
from games import MiniShogiGame


class MiniShogiNNetConv:
    def __init__(self):
        self.args = config.args

        # Neural Net
        input_shape = (
            MiniShogiGame.BOARD_Y, MiniShogiGame.BOARD_X,
            MiniShogiGame.STATE_STACK_HEIGHT)  # s: batch_size x board_x x board_y x moves_count
        action_matrix_shape = MiniShogiGame.ACTION_STACK_HEIGHT * MiniShogiGame.BOARD_Y * MiniShogiGame.BOARD_X

        nn_input = Input(shape=input_shape)

        x = Conv2D(filters=self.args.num_filters, kernel_size=self.args.kernel_size, padding='same',
                   data_format='channels_last', use_bias=False, kernel_regularizer=l2(self.args.reg),
                   name='input_conv')(nn_input)
        x = BatchNormalization(axis=3, name='input_batch_norm')(x)
        x = LeakyReLU(name='input_relu')(x)

        for i in range(self.args.res_layer_num):
            x = self.build_block(x, i + 1)

        v = Conv2D(filters=2, kernel_size=1, padding='same', data_format='channels_last', use_bias=False,
                   kernel_regularizer=l2(self.args.req), name='value_conv')(x)

        v = BatchNormalization(axis=3, name='value_batch_norm')(v)
        v = LeakyReLU(name='value_relu')(v)
        v = Flatten(name='value_flatten')(v)
        v = Dense(1, activation='tanh', name='value_dense_out', use_bias=False)(v)

        pi = Conv2D(filters=85, kernel_size=1, padding='same', data_format='channels_last', use_bias=False,
                    kernel_regularizer=l2(self.args.req), name='policy_conv')(x)
        pi = BatchNormalization(axis=3, name='policy_batch_norm')(pi)
        pi = LeakyReLU(name='policy_relu')(pi)
        pi = Flatten(name='policy_flatten')(pi)
        pi = Dense(action_matrix_shape, activation='softmax', name='policy_dense_out', use_bias=False)(pi)

        self.model = Model(inputs=nn_input, outputs=[pi, v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(self.args.lr))

    def build_block(self, x, index):
        in_x = x
        x = Conv2D(filters=self.args.num_filters, kernel_size=self.args.kernel_size, padding='same',
                   data_format="channels_last", use_bias=False, kernel_regularizer=l2(self.args.reg),
                   name='block_' + str(index) + '_conv')(x)
        x = BatchNormalization(axis=3, name='block' + str(index) + "_batch_norm")(x)
        x = ReLU(name='block_' + str(index) + '_relu')(x)
        x = Add(name='block_' + str(index) + '_add')([in_x, x])

        return x
