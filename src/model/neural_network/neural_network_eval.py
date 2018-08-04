# coding=utf-8
import csv
import os
import datetime
import numpy as np
import tensorflow as tf
import read_data
import rnn_config as config
import random
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
        decay_function = tf.placeholder('float64', [model_config.time_decay_size])
        intensity = Intensity(model_config)
        mutual_intensity = intensity.mutual_intensity
    model = ProposedModel(model_config=model_config)

    loss, c_pred_list, r_pred_list, mi, time_decay = \
        model(placeholder_x=placeholder_x, placeholder_t=placeholder_t,
              mutual_intensity=mutual_intensity, decay_function=decay_function)

    return placeholder_x, placeholder_t, loss, c_pred_list, r_pred_list, mi, time_decay


def fine_tuning(train_config, node_list, data_object, summary_save_path, mutual_intensity_data, time_decay_data):
    placeholder_x, placeholder_t, loss, c_pred_list, r_pred_list, mi, time_decay = node_list

    if train_config.optimizer == 'Adam':
        with tf.variable_scope('adam_para', reuse=True):
            optimizer = tf.train.AdamOptimizer
    else:
        optimizer = tf.train.GradientDescentOptimizer
    optimize_node = optimizer(train_config.learning_rate).minimize(loss)
    initializer = tf.global_variables_initializer()
    batch_count = data_object.get_batch_count()

    merged_summary = tf.summary.merge_all()
    saver = tf.train.Saver

    train_metric_list = list()
    test_metric_list = list()

    with tf.Session() as sess:
        # TODO Debugger待完成
        # sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'Sunzhoujian:6064')
        # construct summary save path
        train_summary_save_path = os.path.join(summary_save_path, 'train')
        test_summary_save_path = os.path.join(summary_save_path, 'test')
        model_path = os.path.join(summary_save_path, 'model')
        os.makedirs(train_summary_save_path)
        os.makedirs(test_summary_save_path)
        os.makedirs(model_path)

        train_summary = tf.summary.FileWriter(train_summary_save_path, sess.graph)
        test_summary = tf.summary.FileWriter(test_summary_save_path, sess.graph)
        sess.run(initializer)

        for i in range(0, train_config.epoch):
            if i == train_config.epoch/2 or i == train_config.epoch-1:
                save_path = saver.save(sess, os.path.join(model_path, train_config.epoch+'model.ckpt'))
                print("Model saved in path: %s" % save_path)

            max_index = data_object.get_batch_count() + 1
            for j in range(0, max_index):
                # time major
                train_x, train_t = data_object.get_train_next_batch()
                actual_batch_size = len(train_x[0])
                max_time_stamp = len(train_x)

                train_dict = {placeholder_x: train_x, placeholder_t: train_t, mi: mutual_intensity_data,
                              time_decay: time_decay_data}
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
            test_dict = {placeholder_x: test_x, placeholder_t: test_t, mi: mutual_intensity_data,
                         time_decay: time_decay_data}
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
    x_depth = 110
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
    time = datetime.datetime.now().strftime("%H%M%S")
    root_path = os.path.abspath('..\\..\\..') + '\\model_evaluate\\Case1\\'
    optimizer = tf.train.AdamOptimizer
    mutual_intensity_path = os.path.join(root_path, 'mutual_intensity.csv')
    base_intensity_path = os.path.join(root_path, 'base_intensity.csv')
    decay_path = os.path.join(root_path, 'decay_function.csv')
    # TODO Learning rate decay 事实上没有实现
    learning_rate_decay = 0.001
    save_path = root_path+time+"\\"
    epoch = 100

    x_path = os.path.join(root_path, '20180803194219_x.npy')
    t_path = os.path.join(root_path, '20180803194219_t.npy')
    encoding = 'utf-8-sig'

    # random search parameter
    batch_candidate = [64, 128, 256, 512]
    actual_batch_size = batch_candidate[random.randint(0, 3)]
    num_hidden_candidate = [32, 64, 128, 256]
    num_hidden = num_hidden_candidate[random.randint(0, 3)]
    zero_state = np.random.normal(0, 1, [num_hidden, ])
    learning_rate = 10**random.uniform(-3, 0)
    threshold_candidate = [0.3, 0.4, 0.5]
    threshold = threshold_candidate[random.randint(0, 2)]

    model_config = config.ModelConfiguration(x_depth=x_depth, t_depth=t_depth, max_time_stamp=max_time_stamp,
                                             num_hidden=num_hidden, cell_type=cell_type, c_r_ratio=c_r_ratio,
                                             activation=activation, init_strategy=init_map, zero_state=zero_state,
                                             init_map=init_map, batch_size=model_batch_size, threshold=threshold)
    train_config = config.TrainingConfiguration(optimizer=optimizer, learning_rate_decay=learning_rate_decay,
                                                save_path=save_path, actual_batch_size=actual_batch_size, epoch=epoch,
                                                decay_step=decay_step, learning_rate=learning_rate,
                                                mutual_intensity_path=mutual_intensity_path,
                                                base_intensity_path=base_intensity_path, file_encoding=encoding,
                                                t_path=t_path, x_path=x_path, decay_path=decay_path)

    return train_config, model_config


def read_time_decay(path, decay_length):
    time_decay = None
    with open(path, 'r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            time_decay = line
            if len(time_decay) != decay_length:
                raise ValueError('')
            break

    return np.array(time_decay)


def validation_test():
    for i in range(0, 2):
        new_graph = tf.Graph()
        with new_graph.as_default():
            train_config, model_config = config.validate_configuration_set()
            time_decay_data = read_time_decay(train_config.decay_path, model_config.time_decay_size)
            data_object = read_data.LoadData(train_config=train_config, model_config=model_config)
            key_node_list = build_model(model_config)
            mutual_intensity_data = \
                Intensity.read_mutual_intensity(encoding=train_config.encoding,
                                                mutual_intensity_path=train_config.mutual_intensity_path,
                                                size=model_config.input_x_depth)
            fine_tuning(train_config, key_node_list, data_object, train_config.save_path, mutual_intensity_data,
                        time_decay_data)


def main():
    # random parameter search
    for i in range(0, 20):
        new_graph = tf.Graph()
        with new_graph.as_default():
            train_config, model_config = configuration_set()
            time_decay_data = read_time_decay(train_config.decay_path, model_config.time_decay_size)
            data_object = read_data.LoadData(train_config=train_config, model_config=model_config)
            mutual_intensity_data = \
                Intensity.read_mutual_intensity(encoding=train_config.encoding,
                                                mutual_intensity_path=train_config.mutual_intensity_path,
                                                size=model_config.input_x_depth)
            key_node_list = build_model(model_config)
            fine_tuning(train_config, key_node_list, data_object, train_config.save_path, mutual_intensity_data,
                        time_decay_data)


if __name__ == '__main__':
    # validation_test()
    main()
