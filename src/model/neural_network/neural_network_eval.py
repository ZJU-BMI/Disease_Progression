# coding=utf-8
import csv
import random

import numpy as np
import tensorflow as tf

import read_data
import rnn_config as config
from model import ProposedModel
from neural_network import performance_metrics as pm


def build_model(train_config, model_config):
    # input define
    max_time_stamp = model_config.max_time_stamp
    batch_size = train_config.batch_size
    x_depth = model_config.input_x_depth
    t_depth = model_config.input_t_depth

    with tf.name_scope('input'):
        placeholder_x = tf.placeholder('float64', [max_time_stamp, batch_size, x_depth])
        placeholder_t = tf.placeholder('float64', [max_time_stamp, batch_size, t_depth])

    model = ProposedModel(model_config=model_config)

    loss, c_pred_list, r_pred_list, merged_summary = model(input_x=placeholder_x, input_t=placeholder_t)

    return placeholder_x, placeholder_t, loss, c_pred_list, r_pred_list, merged_summary


def fine_tuning(train_config, node_list, data_object, train_save_path, test_save_path):
    placeholder_x, placeholder_t, loss, c_pred_list, r_pred_list, merged_summary = node_list

    optimize_node = train_config.optimizer(train_config.learning_rate).minimize(loss)
    initializer = tf.global_variables_initializer
    batch_count = data_object.get_max_batch_index()

    train_metric_list = list()
    test_metric_list = list()

    with tf.Session() as sess:
        # TODO Debugger待完成
        # sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'Sunzhoujian:6064')
        train_summary = tf.summary.FileWriter(train_save_path, sess.graph)
        test_summary = tf.summary.FileWriter(test_save_path, sess.graph)
        sess.run(initializer)

        for i in range(0, train_config.epoch):
            for j in range(0, train_config):
                # time major
                train_x, train_t = data_object.get_train_next_batch()
                actual_batch_size = len(train_x[0])
                max_time_stamp = len(train_x)

                sess.run([optimize_node], feed_dict={placeholder_x: train_x, placeholder_t: train_t})
                c_pred, r_pred, summary = sess.run([c_pred_list, r_pred_list, merged_summary],
                                                   feed_dict={placeholder_x: train_x, placeholder_t: train_t})
                train_summary.add_summary(summary, i * batch_count + j)
                metric_result = pm.performance_measure(c_pred, r_pred, train_x[1:max_time_stamp],
                                                       train_t[1:max_time_stamp], max_time_stamp - 1, actual_batch_size)
                train_metric_list.append([i, j, metric_result])

                # record metadata
                if i % 4 == 0 and j == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, _, _ = sess.run([c_pred_list, r_pred_list, merged_summary],
                                       feed_dict={placeholder_x: train_x, placeholder_t: train_t},
                                       options=run_options, run_metadata=run_metadata)
                    train_summary.add_run_metadata(run_metadata, 'step%d' % i)

            test_x, test_t = data_object.get_test_data()
            actual_batch_size = len(test_x[0])
            max_time_stamp = len(test_x)
            c_pred, r_pred, summary = sess.run([c_pred_list, r_pred_list, merged_summary],
                                               feed_dict={placeholder_x: test_x, placeholder_t: test_t})
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
    # random search parameter
    num_hidden_candidate = [16, 32, 64, 128, 256]
    num_hidden = num_hidden_candidate[random.randint(0, 4)]
    zero_state = np.random.normal(0, 1, [num_hidden, ])
    learning_rate = 10 ** (random.uniform(-4, 0))

    # fixed parameter
    x_depth = 80
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

    model_config = config.ModelConfiguration(x_depth=x_depth, t_depth=t_depth, max_time_stamp=max_time_stamp,
                                             num_hidden=num_hidden, cell_type=cell_type, c_r_ratio=c_r_ratio,
                                             activation=activation, init_strategy=init_map, zero_state=zero_state,
                                             mutual_intensity_path=mi_path, base_intensity_path=bi_path,
                                             file_encoding=file_encoding, init_map=init_map,
                                             time_decay_function=time_decay_function, threshold=threshold)
    train_config = config.TrainingConfiguration(optimizer=optimizer, train_save_path=train_save_path,
                                                test_save_path=test_save_path, weight_decay=weight_decay,
                                                save_path=save_path, batch_size=batch_size, epoch=epoch)

    return train_config, model_config


def main():
    model_name = 'case'
    configuration_set()

    data_path = ""
    save_path = ""
    time_stamp = ""
    batch_size = ""
    data_object = read_data.LoadData(batch_size, data_path, time_stamp)


if __name__ == '__main__':
    main()
