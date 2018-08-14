# coding=utf-8
import numpy as np
import tensorflow as tf

import revised_rnn_cell as rrc
import rnn_config as config


class RevisedRNN(object):
    def __init__(self, model_configuration):
        """
        :param: model_configuration contain variables as listed
        time_stamp: scalar, length of rnn
        x_depth: scalar, the dimension of x
        t_depth: scalar, the dimension of t, should be 1
        hidden_state: scalar
        init_map: parameter strategy map, at least contain 4 elements with key 'gate_weight',
        'gate_bias', 'candidate_weight', 'candidate_bias'. each key corresponds to a tf.Variable initializer
        activation: string,
        zero_state: the zero state is a tensor with size [hidden_state]
        cell_type: default revised_gru
        """
        self.__time_stamp = model_configuration.max_time_stamp
        self.__x_depth = model_configuration.input_x_depth
        self.__t_depth = model_configuration.input_t_depth
        self.__hidden_state = model_configuration.num_hidden
        self.__init_map = model_configuration.init_map
        self.__zero_state = tf.convert_to_tensor(model_configuration.zero_state, tf.float64)
        self.__activation = model_configuration.activation
        self.__cell_type = model_configuration.cell_type

        # TODO 如果今后有时间，继续把LSTM等其他Gated Cell修改为可以直接输入Time的形式
        # 如果真的这么做了，则需要抽象出一个revised rnn cell的基类，对revised rnn提供统一接口，
        # 留待以后有空了做
        if self.__cell_type != 'revised_gru':
            raise ValueError('other type unsupported yet')
        else:
            with tf.name_scope('RRNN_input'):
                self.rnn_cell = rrc.RevisedGRUCell(self.__hidden_state, self.__init_map, 'rgru_cell', self.__activation)

        print('initialize rnn and build network accomplished')

    def __call__(self, input_x, input_t):
        """
        :param input_x: a tf.placeholder of input_x, with size [time_stamp, batch_size, x_depth] dtype=tf.float64
        :param input_t: a tf.placeholder of input_t, with size [time_stamp, batch_size, t_depth] dtype=tf.float64
                batch_size can be None
        :return: a tensor consists of all hidden states with size [time_stamp, batch_size, hidden_state]
        """
        self.__argument_validation(input_x, input_t)
        state = self.__zero_state
        states_list = list()

        with tf.name_scope('input_unstack'):
            input_x = tf.unstack(input_x, axis=0, name='unstack_x')
            input_t = tf.unstack(input_t, axis=0, name='unstack_t')

        with tf.name_scope('rnn_states'):
            for i in range(0, self.__time_stamp):
                with tf.name_scope('state_' + str(i)):
                    step_i_x = input_x[i]
                    step_i_t = input_t[i]
                    state = self.rnn_cell(input_x=step_i_x, input_t=step_i_t, previous_sate=state)
                states_list.append(state)
            states_tensor = tf.convert_to_tensor(states_list, dtype=tf.float64)
        return states_tensor

    def __argument_validation(self, input_x, input_t):
        pass_flag = True
        incomplete_argument = ""
        if input_x is None:
            pass_flag = False
            incomplete_argument += 'input_x, '
        if input_t is None:
            pass_flag = False
            incomplete_argument += 'input_t, '

        if not pass_flag:
            raise ValueError('argument incomplete : ' + incomplete_argument +
                             " please make sure all args name match the requirement")

        input_x_shape = input_x.shape
        input_t_shape = input_t.shape

        if len(input_x_shape.dims) != 3 or input_x_shape[0].value != self.__time_stamp or \
                input_x_shape[2] != self.__x_depth:
            raise ValueError('input_x shape error')
        if len(input_t_shape.dims) != 3 or input_t_shape[0].value != self.__time_stamp or \
                input_t_shape[2] != self.__t_depth:
            raise ValueError('input_y shape error')
        if input_x_shape[1].value != input_t_shape[1].value:
            raise ValueError('the batch size of input_y, input_x should be same')


def unit_test():
    train_config, model_config = config.validate_configuration_set()

    # feed data with different batch_size
    x_1 = np.random.random_integers(0, 1, [model_config.max_time_stamp, 3, model_config.input_x_depth])
    t_1 = np.random.random_integers(0, 1, [model_config.max_time_stamp, 3, model_config.input_t_depth])
    x_2 = np.random.random_integers(0, 1, [model_config.max_time_stamp, 5, model_config.input_x_depth])
    t_2 = np.random.random_integers(0, 1, [model_config.max_time_stamp, 5, model_config.input_t_depth])

    batch_size = None
    revised_rnn = RevisedRNN(model_configuration=model_config)

    placeholder_x = tf.placeholder('float64', [model_config.max_time_stamp, batch_size, model_config.input_x_depth])
    placeholder_t = tf.placeholder('float64', [model_config.max_time_stamp, batch_size, model_config.input_t_depth])
    state_tensor = revised_rnn(input_x=placeholder_x, input_t=placeholder_t)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(state_tensor, feed_dict={placeholder_x: x_1, placeholder_t: t_1})
        sess.run(state_tensor, feed_dict={placeholder_x: x_2, placeholder_t: t_2})


if __name__ == '__main__':
    unit_test()
