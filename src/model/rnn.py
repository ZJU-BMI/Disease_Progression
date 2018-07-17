import numpy as np
import tensorflow as tf

import revised_rnn_cell


class RevisedRNN(object):
    def __init__(self, time_stamp, batch_size, x_depth, t_depth, hidden_state, init_strategy_map,
                 activation, zero_state, name):
        self.time_stamp = time_stamp
        self.batch_size = batch_size
        self.x_depth = x_depth
        self.t_depth = t_depth
        self.hidden_state = hidden_state
        self.init_strategy_map = init_strategy_map
        self.zero_state = zero_state
        self.activation = activation
        self.name = name

        self.rnn_cell = None
        self.input_x = None
        self.input_t = None
        self.states_tensor = None

        self.build()
        print('initialize rnn and build network accomplished')

    def build(self):
        with tf.name_scope('RRNN_input'):
            self.rnn_cell = revised_rnn_cell.RevisedGRUCell(self.hidden_state, self.init_strategy_map, name='RGRU_Cell',
                                                            activation=self.activation)
            self.rnn_cell.build(x_depth=self.x_depth, t_depth=self.t_depth)

            self.input_x = tf.placeholder(dtype=tf.float64, shape=[self.time_stamp, self.batch_size, self.x_depth],
                                          name='input_x')
            self.input_t = tf.placeholder(dtype=tf.float64, shape=[self.time_stamp, self.batch_size, self.t_depth],
                                          name='input_t')

    def __call__(self, *args, **kwargs):
        with tf.name_scope('input_reconstruct'):
            input_x = tf.convert_to_tensor(kwargs['input_x'], dtype=tf.float64, name='unstack_x')
            input_t = tf.convert_to_tensor(kwargs['input_t'], dtype=tf.float64, name='unstack_t')
            self.__argument_validation(input_x, input_t)
            input_x = tf.unstack(input_x, axis=0)
            input_t = tf.unstack(input_t, axis=0)

        state = self.zero_state
        states_list = []
        with tf.name_scope('states_calc'):
            for i in range(0, self.time_stamp):
                step_i_x = input_x[i]
                step_i_t = input_t[i]
                state, _ = self.rnn_cell(step_i_x, step_i_t, state)
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

        if len(input_x_shape.dims) != 3 or input_x_shape[0].value != self.time_stamp or input_x_shape[1].value != \
                self.batch_size or input_x_shape[2] != self.x_depth:
            raise ValueError('input_x shape error')
        if len(input_t_shape.dims) != 3 or input_t_shape[0].value != self.time_stamp or input_t_shape[1].value != \
                self.batch_size or input_t_shape[2] != self.t_depth:
            raise ValueError('input_y shape error')


def main():
    save_path = "D:\\PythonProject\\DiseaseProgression\\src\\model\\train"
    init_map = dict()
    init_map['gate_weight'] = tf.random_normal_initializer(0, 1)
    init_map['gate_bias'] = tf.random_normal_initializer(0, 1)
    init_map['candidate_weight'] = tf.random_normal_initializer(0, 1)
    init_map['candidate_bias'] = tf.random_normal_initializer(0, 1)

    x = np.random.normal(0, 1, [3, 2, 4], )
    t = np.random.normal(0, 1, [3, 2, 1])

    zero_state = np.random.normal(0, 1, [5, ])
    revised_rnn = RevisedRNN(time_stamp=3, batch_size=2, x_depth=4, t_depth=1, hidden_state=5,
                             init_strategy_map=init_map, activation=tf.tanh, zero_state=zero_state, name='rnn')
    state = revised_rnn(input_x=revised_rnn.input_x, input_t=revised_rnn.input_t)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(state, feed_dict={revised_rnn.input_x: x, revised_rnn.input_t: t})
        tf.summary.FileWriter(save_path, sess.graph)


if __name__ == '__main__':
    main()
