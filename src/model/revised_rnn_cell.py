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

        # define weight and input shape
        self.parameter_map = {}
        self.t_depth = None
        self.x_depth = None

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
        with tf.variable_scope("cell_para", reuse=tf.AUTO_REUSE):
            self.parameter_map['gate_weight'] = tf.get_variable(name='gate_weight',
                                                                shape=[t_depth, self._hidden_state * 2],
                                                                initializer=self._initial_strategy_map['gate_weight'],
                                                                dtype=tf.float64)
            self.parameter_map['gate_bias'] = tf.get_variable(name='gate_bias', shape=[self._hidden_state * 2],
                                                              initializer=self._initial_strategy_map['gate_bias'],
                                                              dtype=tf.float64)
            self.parameter_map['candidate_weight'] = tf.get_variable(name='candidate_weight',
                                                                     shape=
                                                                     [x_depth + self._hidden_state, self._hidden_state],
                                                                     initializer=
                                                                     self._initial_strategy_map['candidate_weight'],
                                                                     dtype=tf.float64)
            self.parameter_map['candidate_bias'] = tf.get_variable(name='candidate_bias', shape=[self._hidden_state],
                                                                   initializer=
                                                                   self._initial_strategy_map['candidate_bias'],
                                                                   dtype=tf.float64)
        self.x_depth = x_depth
        self.t_depth = t_depth
        self.built = True

    def __call__(self, input_x, input_t, state):
        """
        :param input_x:
        :param input_t:
        x_input is a matrix with shape [batch_size, input_depth]
        t_input is a matrix with shape [batch_size, 1]
        :param state:
        :return:
        """
        with tf.variable_scope("cell_para", reuse=tf.AUTO_REUSE):
            gate_weight = tf.get_variable("gate_weight", shape=[self.t_depth, self._hidden_state * 2], dtype=tf.float64)
            gate_bias = tf.get_variable("gate_bias", shape=[self._hidden_state * 2], dtype=tf.float64)
            candidate_weight = tf.get_variable("candidate_weight",
                                               shape=[self.x_depth + self._hidden_state, self._hidden_state],
                                               dtype=tf.float64)
            candidate_bias = tf.get_variable("candidate_bias", shape=[self._hidden_state], dtype=tf.float64)

        with tf.name_scope('gate_calc'):
            gate_value = math_ops.matmul(input_t, gate_weight)
            gate_value = nn_ops.bias_add(gate_value, gate_bias)
            gate_value = math_ops.sigmoid(gate_value)

        with tf.name_scope('gate_split'):
            r, u = array_ops.split(value=gate_value, num_or_size_splits=2, axis=1)
            r_state = r * state

        with tf.name_scope('candidate_calc'):
            concat = array_ops.concat([input_x, r_state], axis=1)
            candidate = math_ops.matmul(concat, candidate_weight, name='candidate_matmul')
            candidate = nn_ops.bias_add(candidate, candidate_bias, name='candidate_bias_add')
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

