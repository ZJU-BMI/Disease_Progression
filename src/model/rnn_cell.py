import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell


class RevisedGRUCell(LayerRNNCell):
    """
    compared to standard lstm, change W*[x, h_t-1] to W*[x, h_t-1, t] to import t
    using peephole connection to make full use of time information

    we need to define all tensors' shape explicitly
    """

    def __init__(self, hidden_states, initial_strategy_set, reuse=None, name=None, activation=None):
        """
        :param hidden_states: length of hidden state vector
        :param name: the name in name scope
        :param initial_strategy_set: a dict, contains all parameters initializer
        """
        super(RevisedGRUCell, self).__init__(_reuse=reuse, name=name)
        self.legal_examine(hidden_states, name, initial_strategy_set)
        self._hidden_state = hidden_states
        self._activation = activation
        self.initialized_flag = True
        self._gate_weight = None
        self._gate_bias = None
        self._candidate_weight = None
        self._candidate_bias = None

        self.built = False

        self.legal_examine(hidden_states, name, initial_strategy_set)

    def build(self, inputs_shape):
        x_shape, t_shape = inputs_shape

        if x_shape[1].value is None:
            raise ValueError("Expected x_shape[-1] to be known, saw shape: %s"
                             % x_shape)
        if t_shape[1].value is None:
            raise ValueError("Expected t_shape[-1] to be known, saw shape: %s"
                             % t_shape)

        # define the parameter a GRU will use
        with tf.name_scope('GRU_Cell'):
            with tf.variable_scope('GRU_Cell_Para'):
                self._gate_weight = tf.get_variable(name='gate_weight',
                                                    shape=[self._hidden_state * 2],
                                                    initializer=self.initial_strategy_set['gate_weight'])
                self._gate_bias = tf.get_variable(name='gate_bias',
                                                  shape=[self._hidden_state * 2],
                                                  initializer=self.initial_strategy_set['gate_bias'])
                self._candidate_weight = tf.get_variable(name='candidate_weight',
                                                         shape=[self._hidden_state],
                                                         initializer=self.initial_strategy_set['candidate_weight'])
                self._candidate_bias = tf.get_variable(name='candidate_bias',
                                                       shape=[self._hidden_state],
                                                       initializer=self.initial_strategy_set['candidate_bias'])
        self.built = True

    def call(self, inputs, **kwargs):
        """
        :param inputs: contain x_n, and t_n
        :param kwargs need contains shape
        :return:
        """
        state = kwargs['state']
        x_input, t_input = inputs

        gate_value = math_ops.matmul(array_ops.concat([t_input], axis=1), self._gate_kernel)
        gate_value = nn_ops.bias_add(gate_value, self._gate_bias)
        gate_value = math_ops.sigmoid(gate_value)

        r, u = array_ops.split(value=gate_value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_h = u * state + (1 - u) * c
        return new_h, new_h

    # todo, exam whether the input are legal
    def legal_examine(self, hidden_states, name, initial_strategy_set):
        # hidden_state, input_size should be the form of [batch_size, num]
        # all parameter should be initialized by explicitly defined initializer
        pass
