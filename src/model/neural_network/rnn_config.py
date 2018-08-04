# coding=utf-8
import datetime
import os

import numpy as np
import tensorflow as tf


class ModelConfiguration(object):
    def __init__(self, x_depth, max_time_stamp, num_hidden, cell_type, init_map, batch_size,
                 c_r_ratio, activation, init_strategy, zero_state, t_depth, threshold, time_decay_size=10000):
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
        :param time_decay_size: which is a scalar, default 10000
        intensity of corresponding time stamps
        :param threshold: threshold for metrics
        """

        self.time_decay_size = time_decay_size

        # Model Parameters
        self.c_r_ratio = c_r_ratio
        self.input_x_depth = x_depth
        self.input_t_depth = t_depth
        self.batch_size = batch_size
        self.max_time_stamp = max_time_stamp

        # Network Parameters
        self.num_hidden = num_hidden
        self.cell_type = cell_type
        self.activation = activation
        self.zero_state = zero_state
        self.init_strategy = init_strategy

        # parameter initializer
        self.init_map = init_map
        self.threshold = threshold
        self.__meta_data = self.__write_meta_data()

    def __write_meta_data(self):
        meta_data = dict()
        meta_data['time_decay_function'] = self.time_decay_size
        meta_data['c_r_ratio'] = self.c_r_ratio
        meta_data['input_x_depth'] = self.input_x_depth
        meta_data['input_t_depth'] = self.input_t_depth
        meta_data['max_time_stamp'] = self.max_time_stamp
        meta_data['num_hidden'] = self.num_hidden
        meta_data['cell_type'] = self.cell_type
        meta_data['activation'] = self.activation
        meta_data['batch_size'] = self.batch_size
        meta_data['zero_state'] = self.zero_state
        meta_data['init_strategy'] = self.init_strategy
        meta_data['init_map'] = self.init_map
        meta_data['threshold'] = self.threshold
        return meta_data

    @property
    def meta_data(self):
        return self.__meta_data


class TrainingConfiguration(object):
    def __init__(self, optimizer, learning_rate, learning_rate_decay, save_path, actual_batch_size, decay_step, epoch,
                 mutual_intensity_path, base_intensity_path, file_encoding, x_path, t_path, decay_path):
        """
        :param optimizer:
        :param learning_rate:
        :param learning_rate_decay:
        :param save_path:
        :param actual_batch_size:
        :param decay_step:
        :param epoch:
        :param mutual_intensity_path: a file path, reading the information of mutual intensity
        :param base_intensity_path: a file path, reading the information of base intensity
        """
        self.learning_rate = learning_rate
        self.decay_step = decay_step
        self.optimizer = optimizer
        self.learning_rate_decay = learning_rate_decay
        self.save_path = save_path
        self.actual_batch_size = actual_batch_size
        self.epoch = epoch
        self.mutual_intensity_path = mutual_intensity_path
        self.base_intensity_path = base_intensity_path
        self.encoding = file_encoding
        self.x_path = x_path
        self.t_path = t_path
        self.decay_path = decay_path
        self.meta_data = self.set_meta_data()

    def set_meta_data(self):
        meta_data = dict()
        meta_data['learning_rate'] = self.learning_rate
        meta_data['decay_step'] = self.decay_step
        meta_data['optimizer'] = self.optimizer
        meta_data['learning_rate_decay'] = self.learning_rate_decay
        meta_data['mutual_intensity_path'] = self.mutual_intensity_path
        meta_data['base_intensity_path'] = self.base_intensity_path
        meta_data['save_path'] = self.save_path
        meta_data['actual_batch_size'] = self.actual_batch_size
        meta_data['epoch'] = self.epoch
        meta_data['encoding'] = self.encoding
        meta_data['x_path'] = self.x_path
        meta_data['t_path'] = self.t_path

        return meta_data


def validate_configuration_set():
    # fixed model parameters
    x_depth = 20
    t_depth = 1
    max_time_stamp = 5
    cell_type = 'revised_gru'
    c_r_ratio = 1
    activation = 'tanh'
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
    decay_step = 100
    model_batch_size = None

    # fixed train parameters
    now_time = datetime.datetime.now().strftime('%H%M%S')
    all_path = os.path.abspath('..\\..\\..') + '\\model_evaluate\\ValidationTest\\'
    optimizer = 'default'
    mutual_intensity_path = os.path.join(all_path, 'mutual_intensity.csv')
    base_intensity_path = os.path.join(all_path, 'base_intensity.csv')
    learning_rate_decay = 0.001
    actual_batch_size = 16

    save_path = all_path + now_time + "\\"
    os.makedirs(save_path)

    epoch = 3
    time_decay_size = 10000
    x_path = os.path.join(all_path, '20180803194219_x.npy')
    t_path = os.path.join(all_path, '20180803194219_t.npy')
    decay_path = os.path.join(all_path, '20180803194219_x.npy')
    encoding = 'utf-8-sig'

    # random search parameter
    num_hidden = 16
    zero_state = np.random.normal(0, 1, [num_hidden, ])
    learning_rate = 0.001
    threshold = 0.5

    model_config = ModelConfiguration(x_depth=x_depth, t_depth=t_depth, max_time_stamp=max_time_stamp,
                                      num_hidden=num_hidden, cell_type=cell_type, c_r_ratio=c_r_ratio,
                                      activation=activation, init_strategy=init_map, zero_state=zero_state,
                                      init_map=init_map, batch_size=model_batch_size,
                                      time_decay_size=time_decay_size, threshold=threshold, )
    train_config = TrainingConfiguration(optimizer=optimizer, learning_rate_decay=learning_rate_decay,
                                         save_path=save_path, actual_batch_size=actual_batch_size, epoch=epoch,
                                         decay_step=decay_step, learning_rate=learning_rate,
                                         mutual_intensity_path=mutual_intensity_path,
                                         base_intensity_path=base_intensity_path, file_encoding=encoding,
                                         t_path=t_path, x_path=x_path, decay_path=decay_path)

    return train_config, model_config


if __name__ == "__main__":
    validate_configuration_set()
