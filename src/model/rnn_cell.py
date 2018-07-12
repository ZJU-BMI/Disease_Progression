import tensorflow as tf
from tensorflow.python.ops import array_ops


class LSTMRevised1Cell(object):
    """
    compared to standard lstm, change W*[x, h_t-1] to W*[x, h_t-1, t] to import t
    using peephole connection to make full use of time information
    """

    def __init__(self, input_size, hidden_states, name, initial_strategy_set, use_peephole=True,
                 forget_bias=1.0):
        """
        :param input_size: length of each input vector
        :param hidden_states: length of hidden state vector
        :param name: the name in name scope
        :param initial_strategy_set: a dict, contains all parameters initializer
        :param use_peephole: using peephole connection or not, default true to make full use of time information
        :param forget_bias: default 1.0
        """
        self.legal_examine(input_size, hidden_states, name, initial_strategy_set, use_peephole, forget_bias)
        with tf.name_scope(name=name):
            with tf.variable_scope('revised lstm parameter'):
                self._num_inputs = input_size
                self._hidden_state = hidden_states
                self._forget_bias = forget_bias
                self._use_peephole = use_peephole
                self._weight = tf.get_variable('weight', [input_size + hidden_states + 1, hidden_states * 4],
                                               initializer=initial_strategy_set['weight'])
                self._bias = tf.get_variable('bias', [hidden_states * 4], initializer=initial_strategy_set['bias'])

                if use_peephole:
                    self._w_i = tf.get_variable('wci', [hidden_states], initializer=initial_strategy_set['wci'])
                    self._w_f = tf.get_variable('wcf', [hidden_states], initializer=initial_strategy_set['wcf'])
                    self._w_o_diag = tf.get_variable('wco', [hidden_states], initializer=initial_strategy_set['wco'])
                else:
                    self._w_i = array_ops.zeros([hidden_states])
                    self._w_f = array_ops.zeros([hidden_states])
                    self._w_o = array_ops.zeros([hidden_states])
        self.initialized_flag = True

    # todo, exam whether the input are legal
    def legal_examine(self, input_size, hidden_states, name, initial_strategy_set, use_peephole, forget_bias):
        # hidden_state, input_size should be the form of [batch_size, num]
        pass

    def init(self):
        pass

    def call(self, input_x, input_t, pre_h_state, pre_cell_state):
        """
        :param input_x:
        :param input_t: vector match shape
        :param pre_h_state:
        :param pre_cell_state:
        :return:
        """
        w_i = self._w_i
        w_f = self._w_f
        w_o = self._w_o
        forget_bias = self._forget_bias

        # todo, 确认拼接是否正确
        x_h = tf.concat([input_x, pre_h_state, input_t], axis=1)
        kernel = tf.nn.sigmoid(tf.matmul(x_h, self._weight), name='get_new_state_info')

        # TODO 留待继续完成


class BlockLSTM2(object):
    """
    directly using time information to decide value of forget gate
    """
