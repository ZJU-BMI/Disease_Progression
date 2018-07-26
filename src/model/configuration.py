# coding=utf-8
import datetime
import os

import numpy as np
import tensorflow as tf


class ModelConfiguration(object):
    def __init__(self, x_depth, max_time_stamp, num_hidden, cell_type, init_map,
                 c_r_ratio, activation, init_strategy, mutual_intensity_path, base_intensity_path, zero_state,
                 file_encoding, time_decay_function, t_depth, threshold):
        """
        :param x_depth: defines the dimension of the input_x in a specific time stamp, it also indicates the number
        of type of event
        :param t_depth: defines the time of a specific time stamp, raise error if it is not 1
        :param max_time_stamp: should be a scalar, the length of RNN
        :param num_hidden: should be a scalar, the dimension of a hidden state
        :param cell_type: should be a string, 'revised_gru' or 'gru'
        :param c_r_ratio: should be a scalar, the coefficient to adjust the weight between classification task and
        regression task.
        :param zero_state: the zero state of rnn, np.ndarray with shape [num_hidden,]
        :param activation: should be a function object, activation function of RNN
        :param init_strategy: parameter initialize strategy for every parameter
        :param mutual_intensity_path: a file path, reading the information of mutual intensity
        :param base_intensity_path: a file path, reading the information of base intensity
        :param file_encoding: intensity file encoding
        :param time_decay_function: which is long (at least 10,000 elements) 1-d np.ndarray, each entry indicates the
        intensity of corresponding time stamps
        :param threshold: threshold for metrics
        """

        self.time_decay_function = time_decay_function

        # Model Parameters
        self.c_r_ratio = c_r_ratio
        self.input_x_depth = x_depth
        self.input_t_depth = t_depth
        self.max_time_stamp = max_time_stamp

        # Network Parameters
        self.num_hidden = num_hidden
        self.cell_type = cell_type
        self.activation = activation
        self.zero_state = zero_state
        self.init_strategy = init_strategy

        # Attention Parameters
        self.mutual_intensity_path = mutual_intensity_path
        self.base_intensity_path = base_intensity_path
        self.file_encoding = file_encoding

        # parameter initializer
        self.init_map = init_map
        self.threshold = threshold


class TrainingConfiguration(object):
    def __init__(self, learning_rate, optimizer, weight_decay, train_save_path, test_save_path, batch_size, iteration):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.train_save_path = train_save_path
        self.test_save_path = test_save_path
        self.batch_size = batch_size
        self.iteration = iteration


class TestConfiguration(object):
    root_path = os.path.abspath('..\\..')

    @staticmethod
    def get_test_model_config():
        # model config
        num_hidden = 3
        x_depth = 6
        t_depth = 1
        max_time_stamp = 4
        cell_type = 'revised_gru'
        zero_state = np.random.normal(0, 1, [num_hidden, ])
        activation = tf.tanh
        init_map = dict()
        init_map['gate_weight'] = tf.random_normal_initializer(0, 1)
        init_map['gate_bias'] = tf.random_normal_initializer(0, 1)
        init_map['candidate_weight'] = tf.random_normal_initializer(0, 1)
        init_map['candidate_bias'] = tf.random_normal_initializer(0, 1)
        init_map['classification_weight'] = tf.random_normal_initializer(0, 1)
        init_map['classification_bias'] = tf.random_normal_initializer(0, 1)
        init_map['regression_weight'] = tf.random_normal_initializer(0, 1)
        init_map['regression_bias'] = tf.random_normal_initializer(0, 1)
        init_map['mutual_intensity'] = tf.random_normal_initializer(0, 1)
        init_map['base_intensity'] = tf.random_normal_initializer(0, 1)
        init_map['mutual_intensity'] = tf.random_normal_initializer(0, 1)
        init_map['combine'] = tf.random_normal_initializer(0, 1)
        mi_path = TestConfiguration.root_path + "\\resource\\mutual_intensity_sample.csv"
        bi_path = TestConfiguration.root_path + "\\resource\\base_intensity_sample.csv"
        file_encoding = 'utf-8-sig'
        c_r_ratio = 1
        threshold = 0.2
        # time decay由于日期是离散的，每一日的强度直接采用硬编码的形式写入
        time_decay_function = np.random.normal(0, 1, [10000, ])

        model_config = ModelConfiguration(x_depth=x_depth, t_depth=t_depth, max_time_stamp=max_time_stamp,
                                          num_hidden=num_hidden, cell_type=cell_type, c_r_ratio=c_r_ratio,
                                          activation=activation, init_strategy=init_map, zero_state=zero_state,
                                          mutual_intensity_path=mi_path, base_intensity_path=bi_path,
                                          file_encoding=file_encoding, init_map=init_map,
                                          time_decay_function=time_decay_function, threshold=threshold)
        return model_config

    @staticmethod
    def get_test_training_config():
        # training configuration
        learning_rate = 0.1
        optimizer = tf.train.AdamOptimizer
        weight_decay = 0.0001

        now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        train_save_path = TestConfiguration.root_path + '\\model_evaluate\\train\\' + now_time + "\\"
        test_save_path = TestConfiguration.root_path + '\\model_evaluate\\test\\' + now_time + "\\"
        os.makedirs(train_save_path)
        os.makedirs(test_save_path)
        batch_size = None
        iteration = 20

        train_config = TrainingConfiguration(learning_rate=learning_rate, optimizer=optimizer,
                                             weight_decay=weight_decay, train_save_path=train_save_path,
                                             test_save_path=test_save_path, batch_size=batch_size,
                                             iteration=iteration)
        return train_config
