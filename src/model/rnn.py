# coding=utf-8
import os

import numpy as np
import tensorflow as tf

import revised_rnn_cell as rrc


class RevisedRNN(object):
    def __init__(self, time_stamp, x_depth, t_depth, hidden_state, init_map, activation, zero_state, cell_type):
        """
        :param time_stamp: scalar, length of rnn
        :param x_depth: scalar, the dimension of x
        :param t_depth: scalar, the dimension of t, should be 1
        :param hidden_state: scalar
        :param init_map: parameter strategy map, at least contain 4 elements with key 'gate_weight',
        'gate_bias', 'candidate_weight', 'candidate_bias'. each key corresponds to a tf.Variable initializer
        :param activation: a function object
        :param zero_state: the zero state of rnn with size [hidden_state]
        :param cell_type: default revised_gru
        """
        self.__time_stamp = time_stamp
        self.__x_depth = x_depth
        self.__t_depth = t_depth
        self.__hidden_state = hidden_state
        self.__init_map = init_map
        self.__zero_state = zero_state
        self.__activation = activation

        # TODO 如果今后有时间，继续把LSTM等其他Gated Cell修改为可以直接输入Time的形式
        # 如果真的这么做了，则需要抽象出一个revised rnn cell的基类，对revised rnn提供统一接口，
        # 留待以后有空了做
        if cell_type != 'revised_gru':
            raise ValueError('other type unsupported yet')
        self.__rnn_cell = cell_type
        self.__build()

        print('initialize rnn and build network accomplished')

    def __build(self):
        with tf.name_scope('RRNN_input'):
            self.rnn_cell = rrc.RevisedGRUCell(self.__hidden_state, self.__init_map, 'rgru_cell', self.__activation)

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
            self.states_tensor = states_tensor
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
    root_path = os.path.abspath('..\\..')
    save_path = root_path + "\\src\\model\\train"

    time_stamp = 5
    batch_size = None
    x_depth = 4
    t_depth = 1
    hidden_state = 16
    placeholder_x = tf.placeholder(name='x', shape=[time_stamp, batch_size, x_depth], dtype=tf.float64)
    placeholder_t = tf.placeholder(name='x', shape=[time_stamp, batch_size, t_depth], dtype=tf.float64)
    zero_state = tf.convert_to_tensor(np.random.normal(0, 1, [hidden_state]))
    init = dict()
    init['gate_weight'] = tf.random_normal_initializer(0, 1)
    init['gate_bias'] = tf.random_normal_initializer(0, 1)
    init['candidate_weight'] = tf.random_normal_initializer(0, 1)
    init['candidate_bias'] = tf.random_normal_initializer(0, 1)

    # feed data with different batch_size
    x_1 = np.random.random_integers(0, 1, [time_stamp, 3, x_depth])
    t_1 = np.random.random_integers(0, 1, [time_stamp, 3, t_depth])
    x_2 = np.random.random_integers(0, 1, [time_stamp, 5, x_depth])
    t_2 = np.random.random_integers(0, 1, [time_stamp, 5, t_depth])

    revised_rnn = RevisedRNN(time_stamp=time_stamp, x_depth=x_depth, t_depth=t_depth, hidden_state=hidden_state,
                             init_map=init, activation=tf.tanh, zero_state=zero_state, cell_type='revised_gru')

    state_tensor = revised_rnn(input_x=placeholder_x, input_t=placeholder_t)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(state_tensor, feed_dict={placeholder_x: x_1, placeholder_t: t_1})
        sess.run(state_tensor, feed_dict={placeholder_x: x_2, placeholder_t: t_2})
        tf.summary.FileWriter(save_path, sess.graph)


if __name__ == '__main__':
    unit_test()
