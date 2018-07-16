import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


class RevisedGRUCell(object):
    """
    in the consideration of compatibility, all vector should be defined as matrix with shape= (k,)
    the input_x shape should be 2-d matrix with shape (time stamp, input_depth)
    the input_t shape should be 2-d matrix with shape (time stamp, 1)

    revised the method that calculate gate
    """

    def __init__(self, hidden_states, initial_strategy_map, name=None, activation=None):
        """
        :param hidden_states: length of hidden state vector
        :param name: the name in name scope
        :param initial_strategy_map: a dict, contains all parameters initializer
        """
        self._hidden_state = hidden_states
        self._activation = activation
        self._initial_strategy_map = initial_strategy_map
        self._name = name

        # define weight
        self._gate_weight = None
        self._gate_bias = None
        self._candidate_weight = None
        self._candidate_bias = None

        self.built = False

        self.legal_examine()

    @property
    def state_size(self):
        return self._hidden_state

    @property
    def output_size(self):
        return self._hidden_state

    def build(self, x_depth, t_depth):
        """

        :param x_depth:
        :param t_depth:
        :return:
        """

        # define the parameter a GRU will use
        with tf.name_scope(self._name):
            with tf.variable_scope(self._name + "_para", reuse=tf.AUTO_REUSE):
                self._gate_weight = tf.get_variable(name='gate_weight',
                                                    shape=[t_depth, self._hidden_state * 2],
                                                    initializer=self._initial_strategy_map['gate_weight'],
                                                    dtype=tf.float64)
                self._gate_bias = tf.get_variable(name='gate_bias',
                                                  shape=[self._hidden_state * 2],
                                                  initializer=self._initial_strategy_map['gate_bias'],
                                                  dtype=tf.float64)
                self._candidate_weight = tf.get_variable(name='candidate_weight',
                                                         shape=[x_depth + self._hidden_state, self._hidden_state],
                                                         initializer=self._initial_strategy_map['candidate_weight'],
                                                         dtype=tf.float64)
                self._candidate_bias = tf.get_variable(name='candidate_bias',
                                                       shape=[self._hidden_state],
                                                       initializer=self._initial_strategy_map['candidate_bias'],
                                                       dtype=tf.float64)

    def __call__(self, input_x, input_t, state):
        """
        :param input_x:
        :param input_t:
        x_input is a matrix with shape [batch_size, input_depth]
        t_input is a matrix with shape [batch_size, 1]
        :param state:
        :return:
        """
        gate_weight = self._gate_weight
        gate_bias = self._gate_bias
        candidate_weight = self._candidate_weight
        candidate_bias = self._candidate_bias

        with tf.name_scope('gate_calc'):
            gate_value = math_ops.matmul(input_t, gate_weight)
            gate_value = nn_ops.bias_add(gate_value, gate_bias)
            gate_value = math_ops.sigmoid(gate_value)

        with tf.name_scope('gate_split'):
            r, u = array_ops.split(value=gate_value, num_or_size_splits=2, axis=1)
            r_state = r * state

        with tf.name_scope('candidate_calc'):
            concat = array_ops.concat([input_x, r_state], axis=1)
            candidate = math_ops.matmul(concat, candidate_weight)
            candidate = nn_ops.bias_add(candidate, candidate_bias)
            c = self._activation(candidate)

        with tf.name_scope('new_state'):
            new_h = u * state + (1 - u) * c

        return new_h, new_h

    def legal_examine(self):
        """
        This function is used to examine whether the parameter are legal
        """
        legal_flag = True

        hidden_states = self._hidden_state
        initial_strategy_map = self._initial_strategy_map
        name = self._name

        if hidden_states < 0 or not isinstance(hidden_states, int):
            legal_flag = False

        if name is not None and not isinstance(name, str):
            legal_flag = False

        key_name = ['gate_weight', 'gate_bias', 'candidate_weight', 'candidate_bias']
        for key in initial_strategy_map:
            if not (isinstance(initial_strategy_map[key], tf.keras.initializers.Initializer) and (key in key_name)):
                legal_flag = False

        if not legal_flag:
            raise Exception('the format of parameter is not right')


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
        self.output_y_c = None
        self.output_y_r = None

        self.build()
        print('initialize rnn and build network accomplished')

    def build(self):
        with tf.name_scope('RevisedRNN'):
            self.rnn_cell = RevisedGRUCell(self.hidden_state, self.init_strategy_map, name='RGRU_Cell',
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
        return states_list

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
