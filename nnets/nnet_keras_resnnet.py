from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2

import config
from games import MiniShogiGame


class MiniShogiNNetKerasResNet:
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
        x = ReLU(name='input_relu')(x)

        for i in range(self.args.res_layer_num):
            x = self.build_block(x, i + 1)

        x = Flatten(name='out_flatten')(x)
        x = Dense(3200, activation='linear', name='out_dense_1', use_bias=False)(x)
        x = BatchNormalization(axis=1, name='out_batch_norm_1')(x)
        x = ReLU(name='out_relu_1')(x)
        x = Dropout(self.args.dropout, name='out_dropout_1')(x)

        x = Dense(1600, activation='linear', name='out_dense_2', use_bias=False)(x)
        x = BatchNormalization(axis=1, name='out_batch_norm_2')(x)
        x = ReLU(name='out_relu_2')(x)
        x = Dropout(self.args.dropout, name='out_dropout_2')(x)

        x = Dense(800, activation='linear', name='out_dense_3', use_bias=False)(x)
        x = BatchNormalization(axis=1, name='out_batch_norm_3')(x)
        x = ReLU(name='out_relu_3')(x)
        x = Dropout(self.args.dropout, name='out_dropout_3')(x)

        v = Dense(1, activation='tanh', name='value_dense_out', use_bias=False)(x)

        pi = Dense(action_matrix_shape, activation='softmax', name='policy_dense_out', use_bias=False)(x)

        self.model = Model(inputs=nn_input, outputs=[pi, v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(self.args.lr))
        #self.model.summary()

    def build_block(self, x, index):
        in_x = x
        x = Conv2D(filters=self.args.num_filters, kernel_size=self.args.kernel_size, padding='same',
                   data_format="channels_last", use_bias=False, kernel_regularizer=l2(self.args.reg),
                   name='block_' + str(index) + '_conv_1')(x)
        x = BatchNormalization(axis=3, name='block_' + str(index) + "_batch_norm_1")(x)
        x = ReLU(name='block_' + str(index) + '_relu_1')(x)

        x = Conv2D(filters=self.args.num_filters, kernel_size=self.args.kernel_size, padding='same',
                   data_format="channels_last", use_bias=False, kernel_regularizer=l2(self.args.reg),
                   name='block_' + str(index) + '_conv_2')(x)
        x = BatchNormalization(axis=3, name='block_' + str(index) + "_batch_norm")(x)
        x = Add(name='block_' + str(index) + '_add_2')([in_x, x])
        x = ReLU(name='block_' + str(index) + '_relu_2')(x)

        return x
