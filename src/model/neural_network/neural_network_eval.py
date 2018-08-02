# coding=utf-8
import csv
import os

import numpy as np
import tensorflow as tf

import read_data
import rnn_config as config
from intensity import Intensity
from model import ProposedModel
from neural_network import performance_metrics as pm


def build_model(model_config):
    # input define
    max_time_stamp = model_config.max_time_stamp
    batch_size = model_config.batch_size
    x_depth = model_config.input_x_depth
    t_depth = model_config.input_t_depth

    with tf.name_scope('input'):
        placeholder_x = tf.placeholder('float64', [max_time_stamp, batch_size, x_depth])
        placeholder_t = tf.placeholder('float64', [max_time_stamp, batch_size, t_depth])
        intensity = Intensity(model_config)
        mutual_intensity = intensity.mutual_intensity
    model = ProposedModel(model_config=model_config)

    loss, c_pred_list, r_pred_list, merged_summary, mi = \
        model(placeholder_x=placeholder_x, placeholder_t=placeholder_t,
              mutual_intensity=mutual_intensity)

    return placeholder_x, placeholder_t, loss, c_pred_list, r_pred_list, merged_summary, mi


def fine_tuning(train_config, node_list, data_object, summary_save_path, mutual_intensity_data):
    placeholder_x, placeholder_t, loss, c_pred_list, r_pred_list, merged_summary, mi = node_list

    if train_config.optimizer == 'Adam':
        with tf.variable_scope('adam_para', reuse=True):
            optimizer = tf.train.AdamOptimizer
    else:
        optimizer = tf.train.GradientDescentOptimizer
    optimize_node = optimizer(train_config.learning_rate).minimize(loss)
    initializer = tf.global_variables_initializer()
    batch_count = data_object.get_batch_count()

    train_metric_list = list()
    test_metric_list = list()

    with tf.Session() as sess:
        # TODO Debugger待完成
        # sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'Sunzhoujian:6064')
        # construct summary save path
        train_summary_save_path = os.path.join(summary_save_path, 'train')
        test_summary_save_path = os.path.join(summary_save_path, 'test')
        os.makedirs(train_summary_save_path)
        os.makedirs(test_summary_save_path)
        train_summary = tf.summary.FileWriter(train_summary_save_path, sess.graph)
        test_summary = tf.summary.FileWriter(test_summary_save_path, sess.graph)
        sess.run(initializer)

        for i in range(0, train_config.epoch):
            max_index = data_object.get_batch_count() + 1
            for j in range(0, max_index):
                # time major
                train_x, train_t = data_object.get_train_next_batch()
                actual_batch_size = len(train_x[0])
                max_time_stamp = len(train_x)

                train_dict = {placeholder_x: train_x, placeholder_t: train_t, mi: mutual_intensity_data}
                _, c_pred, r_pred, summary = sess.run([optimize_node, c_pred_list, r_pred_list, merged_summary],
                                                      feed_dict=train_dict)
                train_summary.add_summary(summary, i * batch_count + j)
                metric_result = pm.performance_measure(c_pred, r_pred, train_x[1:max_time_stamp],
                                                       train_t[1:max_time_stamp], max_time_stamp - 1, actual_batch_size)
                train_metric_list.append([i, j, metric_result])

                # record metadata
                if i % 4 == 0 and j == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, _, _ = sess.run([c_pred_list, r_pred_list, merged_summary],
                                       feed_dict=train_dict,
                                       options=run_options, run_metadata=run_metadata)
                    train_summary.add_run_metadata(run_metadata, 'step%d' % i)

            test_x, test_t = data_object.get_test_data()
            actual_batch_size = len(test_x[0])
            max_time_stamp = len(test_x)
            test_dict = {placeholder_x: test_x, placeholder_t: test_t, mi: mutual_intensity_data}
            c_pred, r_pred, summary = sess.run([c_pred_list, r_pred_list, merged_summary],
                                               feed_dict=test_dict)
            metric_result = pm.performance_measure(c_pred, r_pred, test_x[1:max_time_stamp], test_t[1:max_time_stamp],
                                                   max_time_stamp - 1, actual_batch_size)
            test_metric_list.append([i, None, metric_result])
            test_summary.add_summary(summary, i * batch_count)

    return train_metric_list, test_metric_list


def write_meta_data(train_meta, model_meta, path):
    with open(path + 'metadata.csv', 'w', encoding='utf-8-sig', newline="") as file:
        csv_writer = csv.writer(file)
        meta_data = []
        for key in train_meta:
            meta_data.append([key, train_meta[key]])
        for key in model_meta:
            meta_data.append([key, model_meta[key]])
        csv_writer.writerow(meta_data)


def configuration_set():
    # fixed model parameters
    x_depth = 20
    t_depth = 1
    max_time_stamp = 5
    cell_type = 'revised_gru'
    c_r_ratio = 1
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
    decay_step = 100
    model_batch_size = None

    # fixed train parameters
    all_path = os.path.abspath('..\\..\\..') + '\\model_evaluate\\ValidationTest\\'
    optimizer = tf.train.AdamOptimizer
    mutual_intensity_path = os.path.join(all_path, 'mutual_intensity.csv')
    base_intensity_path = os.path.join(all_path, 'base_intensity.csv')
    learning_rate_decay = 0.001
    actual_batch_size = 16
    save_path = all_path
    epoch = 3
    time_decay_function = np.random.uniform(0.1, 1, [1, 10000])
    train_x_path = os.path.join(all_path, 'train_input_x.npy')
    train_t_path = os.path.join(all_path, 'train_input_t.npy')
    test_x_path = os.path.join(all_path, 'test_input_x.npy')
    test_t_path = os.path.join(all_path, 'test_input_t.npy')
    encoding = 'utf-8-sig'

    # random search parameter
    num_hidden = 16
    zero_state = np.random.normal(0, 1, [num_hidden, ])
    learning_rate = 0.001
    threshold = 0.5

    model_config = config.ModelConfiguration(x_depth=x_depth, t_depth=t_depth, max_time_stamp=max_time_stamp,
                                             num_hidden=num_hidden, cell_type=cell_type, c_r_ratio=c_r_ratio,
                                             activation=activation, init_strategy=init_map, zero_state=zero_state,
                                             init_map=init_map, batch_size=model_batch_size,
                                             time_decay_function=time_decay_function, threshold=threshold, )
    train_config = config.TrainingConfiguration(optimizer=optimizer, learning_rate_decay=learning_rate_decay,
                                                save_path=save_path, actual_batch_size=actual_batch_size, epoch=epoch,
                                                decay_step=decay_step, learning_rate=learning_rate,
                                                mutual_intensity_path=mutual_intensity_path,
                                                base_intensity_path=base_intensity_path, file_encoding=encoding,
                                                train_t_path=train_t_path, train_x_path=train_x_path,
                                                test_t_path=test_t_path, test_x_path=test_x_path)

    return train_config, model_config


def validation_test():
    for i in range(0, 2):
        new_graph = tf.Graph()
        with new_graph.as_default():
            train_config, model_config = config.validate_configuration_set()
            data_object = read_data.LoadData(train_config=train_config, model_config=model_config)
            key_node_list = build_model(model_config)
            mutual_intensity_data = \
                Intensity.read_mutual_intensity(encoding=train_config.encoding,
                                                mutual_intensity_path=train_config.mutual_intensity_path,
                                                size=model_config.input_x_depth)
            fine_tuning(train_config, key_node_list, data_object, train_config.save_path, mutual_intensity_data)


def main():
    # random parameter search
    for i in range(0, 20):
        new_graph = tf.Graph()
        with new_graph.as_default():
            train_config, model_config = configuration_set()
            data_object = read_data.LoadData(train_config=train_config, model_config=model_config)
            mutual_intensity_data = \
                Intensity.read_mutual_intensity(encoding=train_config.encoding,
                                                mutual_intensity_path=train_config.mutual_intensity_path,
                                                size=model_config.input_x_depth)
            key_node_list = build_model(model_config)
            fine_tuning(train_config, key_node_list, data_object, train_config.save_path, mutual_intensity_data)


if __name__ == '__main__':
    validation_test()
    # main()
