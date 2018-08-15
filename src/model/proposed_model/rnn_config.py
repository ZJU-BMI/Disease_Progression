# coding=utf-8
import datetime
import os

import numpy as np
import tensorflow as tf


class ModelConfiguration(object):
    def __init__(self, x_depth, max_time_stamp, num_hidden, cell_type, init_map, batch_size,
                 c_r_ratio, activation, zero_state, t_depth, threshold, time_decay_size=10000):
        """
        :param x_depth: the dimension of x
        :param max_time_stamp: the max time step of rnn
        :param num_hidden: number of hidden neurons
        :param cell_type: the cell type, should be 'revised_gru'
        :param init_map: the map record all variables' initializer
        :param batch_size: the batch size of model, None is recommended for the flexible of of model
        :param c_r_ratio: the weight of classification task and regression task
        :param activation: the activation function, default 'tanh'
        :param zero_state: the zero state of rnn
        :param t_depth: the dimension of t, default 1
        :param threshold: the calibration
        :param time_decay_size: the length of discrete time decay function, default 10000
        """

        # Model Parameter
        self.c_r_ratio = c_r_ratio
        self.input_x_depth = x_depth
        self.input_t_depth = t_depth
        self.batch_size = batch_size
        self.max_time_stamp = max_time_stamp
        # Network Parameter
        self.num_hidden = num_hidden
        self.cell_type = cell_type
        self.activation = activation
        self.zero_state = zero_state
        # Attention Parameter
        self.time_decay_size = time_decay_size
        # Prediction Parameter
        self.threshold = threshold
        # Parameter Initializer
        self.init_map = init_map

        self.__meta_data = self.__set_meta_data_dict()

    def __set_meta_data_dict(self):
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
        meta_data['init_map'] = self.init_map
        meta_data['threshold'] = self.threshold
        return meta_data

    @property
    def meta_data(self):
        return self.__meta_data


class TrainingConfiguration(object):
    def __init__(self, optimizer, learning_rate, save_path, actual_batch_size, decay_step, epoch,
                 mutual_intensity_path, base_intensity_path, file_encoding, x_path, t_path, decay_path):
        """
        :param optimizer: the Optimizer Object of neural network
        :param learning_rate: initial learn rate
        :param decay_step: the steps that the learning rate decays
        :param epoch: training epoch
        :param save_path: the path that save all summary and result
        :param actual_batch_size: the batch size of data
        :param x_path: the path (including file name) of input x
        :param t_path: the path (including file name) of input t
        :param decay_path: the path (including file name) of discrete time decay function
        :param mutual_intensity_path: the path (including file name) of mutual intensity
        :param base_intensity_path: the path (including file name) of base intensity
        :param file_encoding: the encoding of saved file

        """
        self.learning_rate = learning_rate
        self.decay_step = decay_step
        self.optimizer = optimizer
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
        meta_data['mutual_intensity_path'] = self.mutual_intensity_path
        meta_data['base_intensity_path'] = self.base_intensity_path
        meta_data['decay_path'] = self.decay_path
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
    time_decay_size = 10000
    model_batch_size = None
    init_map = dict()
    init_map['gate_weight'] = tf.contrib.layers.xavier_initializer()
    init_map['candidate_weight'] = tf.contrib.layers.xavier_initializer()
    init_map['classification_weight'] = tf.contrib.layers.xavier_initializer()
    init_map['regression_weight'] = tf.contrib.layers.xavier_initializer()
    init_map['candidate_bias'] = tf.initializers.zeros()
    init_map['classification_bias'] = tf.initializers.zeros()
    init_map['regression_bias'] = tf.initializers.zeros()
    init_map['gate_bias'] = tf.initializers.zeros()
    init_map['mutual_intensity'] = tf.contrib.layers.xavier_initializer()
    init_map['base_intensity'] = tf.contrib.layers.xavier_initializer()
    init_map['combine'] = tf.contrib.layers.xavier_initializer()

    # random search model parameter
    num_hidden = 16
    zero_state = np.zeros([num_hidden, ])
    threshold = 0.5

    # fixed train parameters
    now_time = datetime.datetime.now().strftime('%H%M%S')
    all_path = os.path.abspath('..\\..\\..') + '\\model_evaluate\\ValidationTest\\'
    mutual_intensity_path = os.path.join(all_path, 'mutual_intensity.csv')
    base_intensity_path = os.path.join(all_path, 'base_intensity.csv')
    x_path = os.path.join(all_path, 'validation_x.npy')
    t_path = os.path.join(all_path, 'validation_t.npy')
    decay_path = os.path.join(all_path, 'validation_decay_function.csv')
    save_path = all_path + now_time + "\\"
    os.makedirs(save_path)
    encoding = 'utf-8-sig'
    epoch = 3
    optimizer = 'default'

    # random search train parameter
    learning_rate_decay = 0.001
    decay_step = 100
    learning_rate = 0.001
    actual_batch_size = 16

    model_config = ModelConfiguration(x_depth=x_depth, t_depth=t_depth, max_time_stamp=max_time_stamp,
                                      num_hidden=num_hidden, cell_type=cell_type, c_r_ratio=c_r_ratio,
                                      activation=activation, zero_state=zero_state,
                                      init_map=init_map, batch_size=model_batch_size,
                                      time_decay_size=time_decay_size, threshold=threshold, )
    train_config = TrainingConfiguration(optimizer=optimizer,
                                         save_path=save_path, actual_batch_size=actual_batch_size, epoch=epoch,
                                         decay_step=decay_step, learning_rate=learning_rate,
                                         mutual_intensity_path=mutual_intensity_path,
                                         base_intensity_path=base_intensity_path, file_encoding=encoding,
                                         t_path=t_path, x_path=x_path, decay_path=decay_path)

    return train_config, model_config


if __name__ == "__main__":
    validate_configuration_set()
