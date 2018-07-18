import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


class RevisedGRUCell(object):
    def __init__(self, hidden_states, initial_strategy_map, x_depth, t_depth, name=None, activation=None):
        self.__hidden_state = hidden_states
        self.__activation = activation
        self.__name = name
        self.__initial_strategy_map = initial_strategy_map

        # define parameters
        self.__gate_weight = None
        self.__gate_bias = None
        self.__candidate_weight = None
        self.__candidate_bias = None

        # define weight and input shape
        self.__t_depth = None
        self.__x_depth = None

        self.__legal_examine()

        self.build(x_depth, t_depth)

    @property
    def state_size(self):
        return self.__hidden_state

    @property
    def output_size(self):
        return self.__hidden_state

    def build(self, x_depth, t_depth):
        """
        :param x_depth:
        :param t_depth:
        :return:
        """
        # define the parameter a GRU cell
        with tf.variable_scope("cell_para", reuse=tf.AUTO_REUSE):
            self.__gate_weight = tf.get_variable(name='gate_weight', shape=[t_depth, self.__hidden_state * 2],
                                                 initializer=self.__initial_strategy_map['gate_weight'],
                                                 dtype=tf.float64)
            self.__gate_bias = tf.get_variable(name='gate_bias', shape=[self.__hidden_state * 2],
                                               initializer=self.__initial_strategy_map['gate_bias'], dtype=tf.float64)
            self.__candidate_weight = tf.get_variable(name='candidate_weight',
                                                      shape=[x_depth + self.__hidden_state, self.__hidden_state],
                                                      initializer=self.__initial_strategy_map['candidate_weight'],
                                                      dtype=tf.float64)
            self.__candidate_bias = tf.get_variable(name='candidate_bias', shape=[self.__hidden_state],
                                                    initializer=self.__initial_strategy_map['candidate_bias'],
                                                    dtype=tf.float64)
        self.__x_depth = x_depth
        self.__t_depth = t_depth

    def __call__(self, input_x, input_t, previous_sate):
        """
        :param input_x: tf.placeholder with shape [batch_size, input_depth]
        :param input_t: tf.placeholder a matrix with shape [batch_size, 1]
        :param previous_sate: the previous hidden states
        :return: new hidden state
        """
        with tf.variable_scope("cell_para", reuse=tf.AUTO_REUSE):
            gate_weight = tf.get_variable("gate_weight", shape=[self.__t_depth, self.__hidden_state * 2],
                                          dtype=tf.float64)
            gate_bias = tf.get_variable("gate_bias", shape=[self.__hidden_state * 2], dtype=tf.float64)
            candidate_weight = tf.get_variable("candidate_weight",
                                               shape=[self.__x_depth + self.__hidden_state, self.__hidden_state],
                                               dtype=tf.float64)
            candidate_bias = tf.get_variable("candidate_bias", shape=[self.__hidden_state], dtype=tf.float64)

        with tf.name_scope('gate_calc'):
            gate_value = math_ops.matmul(input_t, gate_weight)
            gate_value = nn_ops.bias_add(gate_value, gate_bias)
            gate_value = math_ops.sigmoid(gate_value)

        with tf.name_scope('gate_split'):
            r, u = array_ops.split(value=gate_value, num_or_size_splits=2, axis=1)
            r = tf.convert_to_tensor(r, name='reset_gate')
            u = tf.convert_to_tensor(u, name='update_gate')
            r_state = r * previous_sate

        with tf.name_scope('candidate_calc'):
            concat = array_ops.concat([input_x, r_state], axis=1)
            candidate = math_ops.matmul(concat, candidate_weight, name='candidate_matmul')
            candidate = nn_ops.bias_add(candidate, candidate_bias, name='candidate_bias_add')
            c = self.__activation(candidate)

        with tf.name_scope('new_state'):
            new_h = u * previous_sate + (1 - u) * c

        return new_h

    def __legal_examine(self):
        """
        This function is used to examine whether the parameter are legal
        """
        legal_flag = True

        hidden_states = self.__hidden_state
        initial_strategy_map = self.__initial_strategy_map
        name = self.__name

        if hidden_states < 0 or not isinstance(hidden_states, int):
            legal_flag = False

        if name is not None and not isinstance(name, str):
            legal_flag = False
        key_name = ['gate_weight', 'gate_bias', 'candidate_weight', 'candidate_bias']
        for key in initial_strategy_map:
            if key in key_name:
                if not (isinstance(initial_strategy_map[key], tf.keras.initializers.Initializer)):
                    legal_flag = False
        if not legal_flag:
            raise Exception('the format of parameter is not right')
