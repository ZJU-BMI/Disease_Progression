import numpy as np
import tensorflow as tf

import revised_rnn_cell


class RevisedRNN(object):
    def __init__(self, time_stamp, batch_size, x_depth, t_depth, hidden_state, init_strategy_map,
                 activation, zero_state, input_x, input_t):
        """
        :param time_stamp: scalar
        :param batch_size: scalar, the number of batch_size, should be assigned explicitly
        :param x_depth: scalar, the dimension of x
        :param t_depth: scalar, the dimension of t, should be 1
        :param hidden_state: scalar
        :param init_strategy_map: parameter strategy map, at least contain 4 elements with key 'gate_weight',
        'gate_bias', 'candidate_weight', 'candidate_bias'. each key corresponds to a tf.Variable initializer
        :param activation: a function object
        :param zero_state: the zero state of rnn
        :param input_x: a tf.placeholder of input_x, with size [time_steps, batch_size, x_depth] dtype=tf.float64
        :param input_t: a tf.placeholder of input_t, with size [time_steps, batch_size, t_depth] dtype=tf.float64
        """
        self.__time_stamp = time_stamp
        self.__batch_size = batch_size
        self.__x_depth = x_depth
        self.__t_depth = t_depth
        self.__hidden_state = hidden_state
        self.__init_strategy_map = init_strategy_map
        self.__zero_state = zero_state
        self.__activation = activation

        # TODO 如果今后有时间，继续把LSTM，或者其他CELL修改为可以直接输入Time的形式
        # 如果真的这么做了，那么__build，call函数都需要大修
        self.__rnn_cell = None
        self.input_x = input_x
        self.input_t = input_t

        self.__build()

        # states tensor is a fully unrolled tensor with shape [time_step, batch_size, hidden_state]
        self.states_tensor = self.__call__(input_x=self.input_x, input_t=self.input_t)
        print('initialize rnn and build network accomplished')

    def __build(self):
        with tf.name_scope('RRNN_input'):
            self.rnn_cell = revised_rnn_cell.RevisedGRUCell(self.__hidden_state, self.__init_strategy_map,
                                                            name='RGRU_Cell', activation=self.__activation,
                                                            x_depth=self.__x_depth, t_depth=self.__t_depth)

    def __call__(self, input_x, input_t):
        self.__argument_validation(input_x, input_t)

        state = self.__zero_state
        states_list = list()

        # zero state append
        with tf.name_scope('zero_state'):
            state_expand = tf.convert_to_tensor(state, dtype=tf.float64)
            state_expand = tf.tile(state_expand, [self.__batch_size])
            state_expand = tf.reshape(state_expand, [self.__batch_size, -1])
        states_list.append(tf.convert_to_tensor(state_expand, dtype=tf.float64))
        with tf.name_scope('rnn_states'):
            input_x = tf.unstack(input_x, axis=0, name='unstack_x')
            input_t = tf.unstack(input_t, axis=0, name='unstack_t')

            for i in range(0, self.__time_stamp - 1):
                with tf.name_scope('revised_gru'):
                    step_i_x = input_x[i]
                    step_i_t = input_t[i]
                    state = self.rnn_cell(step_i_x, step_i_t, state)
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

        if len(input_x_shape.dims) != 3 or input_x_shape[0].value != self.__time_stamp or input_x_shape[1].value != \
                self.__batch_size or input_x_shape[2] != self.__x_depth:
            raise ValueError('input_x shape error')
        if len(input_t_shape.dims) != 3 or input_t_shape[0].value != self.__time_stamp or input_t_shape[1].value != \
                self.__batch_size or input_t_shape[2] != self.__t_depth:
            raise ValueError('input_y shape error')


def main():
    save_path = "D:\\PythonProject\\DiseaseProgression\\src\\model\\train"
    init_map = dict()
    init_map['gate_weight'] = tf.random_normal_initializer(0, 1)
    init_map['gate_bias'] = tf.random_normal_initializer(0, 1)
    init_map['candidate_weight'] = tf.random_normal_initializer(0, 1)
    init_map['candidate_bias'] = tf.random_normal_initializer(0, 1)

    x = np.random.normal(0, 1, [8, 2, 4], )
    t = np.random.normal(0, 1, [8, 2, 1])

    placeholder_x = tf.placeholder(shape=[8, 2, 4], name='input_x', dtype=tf.float64)
    placeholder_t = tf.placeholder(shape=[8, 2, 1], name='input_t', dtype=tf.float64)

    zero_state = np.random.normal(0, 1, [5, ])
    revised_rnn = RevisedRNN(time_stamp=8, batch_size=2, x_depth=4, t_depth=1, hidden_state=5,
                             init_strategy_map=init_map, activation=tf.tanh, zero_state=zero_state,
                             input_t=placeholder_t, input_x=placeholder_x)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(revised_rnn.states_tensor, feed_dict={revised_rnn.input_x: x, revised_rnn.input_t: t})
        tf.summary.FileWriter(save_path, sess.graph)


if __name__ == '__main__':
    main()
