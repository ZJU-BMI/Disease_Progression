import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell


class RevisedGRUCell(LayerRNNCell):
    """
    in the consideration of compatibility, all vector should be defined as matrix with shape= (k,)
    the input_x shape should be 2-d matrix with shape (batch_size, input_depth)
    the input_t shape should be 2-d matrix with shape (batch_size, )

    revised the method that calculate gate
    """

    def __init__(self, hidden_states, initial_strategy_map, reuse=None, name=None, activation=None):
        """
        :param hidden_states: length of hidden state vector
        :param name: the name in name scope
        :param initial_strategy_map: a dict, contains all parameters initializer
        """
        super(RevisedGRUCell, self).__init__(_reuse=reuse, name=name)
        self._hidden_state = hidden_states
        self._activation = activation
        self._initial_strategy_map = initial_strategy_map
        self._reuse = reuse
        self._name = name

        # define weight
        self._gate_weight = None
        self._gate_bias = None
        self._candidate_weight = None
        self._candidate_bias = None

        self.built = False

        self.legal_examine()

    def build(self, inputs_shape):
        """
        :param inputs_shape: contain two elements, inputs_shape_x, inputs_shape_t
        inputs_shape_x is a matrix with shape [batch_size, input_depth]
        inputs_shape_t is a matrix with shape [batch_size, ]
        :return:
        """
        x_shape, t_shape = inputs_shape

        if x_shape[1].value is None:
            raise ValueError("Expected input_x is a two dimensional matrix, but not")

        # define the parameter a GRU will use
        with tf.name_scope('GRUR_Cell'):
            with tf.variable_scope('GRUR_Cell_Para'):
                self._gate_weight = tf.get_variable(name='gate_weight',
                                                    shape=[self._hidden_state * 2],
                                                    initializer=self._initial_strategy_set['gate_weight'])
                self._gate_bias = tf.get_variable(name='gate_bias',
                                                  shape=[self._hidden_state * 2],
                                                  initializer=self._initial_strategy_set['gate_bias'])
                self._candidate_weight = tf.get_variable(name='candidate_weight',
                                                         shape=[self._hidden_state],
                                                         initializer=self._initial_strategy_set['candidate_weight'])
                self._candidate_bias = tf.get_variable(name='candidate_bias',
                                                       shape=[self._hidden_state],
                                                       initializer=self._initial_strategy_set['candidate_bias'])
        self.built = True

    def call(self, inputs, state):
        """
        :param inputs: contain x_n, and t_n
        x_input is a matrix with shape [batch_size, input_depth]
        t_input is a matrix with shape [batch_size, ]
        :param state:
        :return:
        """
        x_input, t_input = inputs

        gate_value = math_ops.matmul(t_input, self._gate_kernel)
        gate_value = nn_ops.bias_add(gate_value, self._gate_bias)
        gate_value = math_ops.sigmoid(gate_value)

        r, u = array_ops.split(value=gate_value, num_or_size_splits=2, axis=1)
        r_state = r * state

        candidate = math_ops.matmul(array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_h = u * state + (1 - u) * c

        return new_h, new_h

    def legal_examine(self):
        """
        This function is used to examine whether the parameter are legal
        """
        legal_flag = True

        hidden_states = self._hidden_state
        activation = self._activation
        initial_strategy_map = self._initial_strategy_map
        reuse = self._reuse
        name = self._name

        if hidden_states < 0 or not isinstance(hidden_states, int):
            legal_flag = False

        if not (activation == 'tanh' or activation == 'relu' or activation == 'sigmoid'):
            legal_flag = False

        if not isinstance(reuse, bool):
            legal_flag = False

        if name is not None and not isinstance(name, str):
            legal_flag = False

        key_name = ['gate_weight', 'gate_bias', 'candidate_weight', 'candidate_bias']
        for key, value in initial_strategy_map:
            if not (isinstance(value, tf.keras.initializers.Initializer) and (key in key_name)):
                legal_flag = False

        if not legal_flag:
            raise Exception('the format of parameter is not right')
