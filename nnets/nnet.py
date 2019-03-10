from keras.models import *
from keras.layers import *
from keras.optimizers import *

import config
from games import MiniShogiGame


class MiniShogiNNet:
    def __init__(self):

        # Neural Net
        self.input_shape = (
            MiniShogiGame.BOARD_Y, MiniShogiGame.BOARD_X,
            MiniShogiGame.STATE_STACK_HEIGHT)  # s: batch_size x board_x x board_y x moves_count
        self.action_matrix_shape = MiniShogiGame.ACTION_STACK_HEIGHT * MiniShogiGame.BOARD_Y * MiniShogiGame.BOARD_X

        # x = Reshape()(self.input_boards)                # batch_size  x board_x x board_y x z
        self.input = Input(shape=self.input_shape)
        # x_input = Reshape((Game.BOARD_Y, Game.BOARD_X, Game.STATE_STACK_HEIGHT, 1))(self.input)
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(config.args.num_channels, config.args.filter_size, padding='same', use_bias=False)(
                self.input)))  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(config.args.num_channels, config.args.filter_size, padding='same', use_bias=False)(
                h_conv1)))  # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(config.args.num_channels, config.args.filter_size, padding='valid', use_bias=False)(
                h_conv2)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(config.args.num_channels, config.args.filter_size, padding='valid', use_bias=False)(
                h_conv3)))  # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(config.args.dropout)(Activation('relu')(
            BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(config.args.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))  # batch_size x 1024
        self.pi = Dense(self.action_matrix_shape, activation='softmax', name='pi')(
            s_fc2)  # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(config.args['lr']))
