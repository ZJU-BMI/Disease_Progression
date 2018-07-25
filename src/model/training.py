# coding=utf-8
import datetime
import os

import numpy as np
import tensorflow as tf

import attention_mechanism
import configuration as config
import intensity
import performance_metrics as pm
import prediction
import revised_rnn


class ModelEvaluate(object):
    def __init__(self, training_config, model_config):
        self.learning_rate = training_config.learning_rate
        self.optimizer = training_config.optimizer
        self.weight_decay = training_config.weight_decay
        self.batch_size = training_config.batch_size
        self.iteration = training_config.iteration
        self.model_config = model_config

        # key node
        self.c_loss = None
        self.r_loss = None
        self.c_pred_list = None
        self.r_pred_list = None
        self.merged_summary = None
        self.placeholder_x = None
        self.placeholder_t = None

    def __call__(self, *args, **kwargs):
        """
        build model
        :param args:
        :param kwargs:
        :return:
        """
        placeholder_x = kwargs['input_x']
        placeholder_t = kwargs['input_t']
        self.placeholder_x = placeholder_x
        self.placeholder_t = placeholder_t

        model_config = self.model_config
        # component define
        revise_gru_rnn = revised_rnn.RevisedRNN(model_configuration=model_config)
        intensity_component = intensity.Intensity(model_configuration=model_config)
        mutual_intensity = intensity_component.mutual_intensity
        attention_model = attention_mechanism.HawkesBasedAttentionLayer(model_configuration=model_config)
        attention_layer = prediction.AttentionMixLayer(model_configuration=model_config,
                                                       mutual_intensity=mutual_intensity,
                                                       revise_rnn=revise_gru_rnn, attention=attention_model)
        prediction_layer = prediction.PredictionLayer(model_configuration=model_config)

        # model construct
        mix_state_list = attention_layer(input_x=placeholder_x, input_t=placeholder_t)
        c_loss, r_loss, c_pred_list, r_pred_list = prediction_layer(mix_hidden_state_list=mix_state_list,
                                                                    input_x=placeholder_x, input_t=placeholder_t)
        # prediction.performance_summary(input_x=placeholder_x, input_t=placeholder_t, c_pred=c_pred_list,
        #                                r_pred=r_pred_list, threshold=model_config.threshold)

        merged_summary = tf.summary.merge_all()

        self.c_loss = c_loss
        self.r_loss = r_loss
        self.c_pred_list = c_pred_list
        self.r_pred_list = r_pred_list
        self.merged_summary = merged_summary
        return c_loss, r_loss, c_pred_list, r_pred_list, merged_summary


def configuration_set():
    root_path = os.path.abspath('..\\..')

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
    mi_path = root_path + "\\resource\\mutual_intensity_sample.csv"
    bi_path = root_path + "\\resource\\base_intensity_sample.csv"
    file_encoding = 'utf-8-sig'
    c_r_ratio = 1
    threshold = 0.2
    # time decay由于日期是离散的，每一日的强度直接采用硬编码的形式写入
    time_decay_function = np.random.normal(0, 1, [10000, ])

    model_config = \
        config.ModelConfiguration(x_depth=x_depth, t_depth=t_depth, max_time_stamp=max_time_stamp,
                                  num_hidden=num_hidden, cell_type=cell_type, c_r_ratio=c_r_ratio,
                                  activation=activation, init_strategy=init_map, zero_state=zero_state,
                                  mutual_intensity_path=mi_path, base_intensity_path=bi_path,
                                  file_encoding=file_encoding, init_map=init_map,
                                  time_decay_function=time_decay_function, threshold=threshold)
    # training configuration
    learning_rate = 0.1
    optimizer = tf.train.AdamOptimizer
    weight_decay = 0.0001

    now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    train_save_path = root_path + '\\model_evaluate\\train\\' + now_time + "\\"
    test_save_path = root_path + '\\model_evaluate\\test\\' + now_time + "\\"
    os.makedirs(train_save_path)
    os.makedirs(test_save_path)
    batch_size = None
    iteration = 20
    train_config = config.TrainingConfiguration(learning_rate=learning_rate, optimizer=optimizer,
                                                weight_decay=weight_decay, train_save_path=train_save_path,
                                                test_save_path=test_save_path, batch_size=batch_size,
                                                iteration=iteration)
    print(train_save_path)
    print(test_save_path)
    return train_config, model_config


def training(train_config, model_config):
    # input define
    max_time_stamp = model_config.max_time_stamp
    batch_size = train_config.batch_size
    x_depth = model_config.input_x_depth
    t_depth = model_config.input_t_depth

    placeholder_x = tf.placeholder('float64', [max_time_stamp, batch_size, x_depth])
    placeholder_t = tf.placeholder('float64', [max_time_stamp, batch_size, t_depth])
    model = ModelEvaluate(training_config=train_config, model_config=model_config)
    c_loss, r_loss, c_pred_list, r_pred_list, merged_summary = model(input_x=placeholder_x, input_t=placeholder_t)
    loss_sum = c_loss + model_config.c_r_ratio * r_loss
    optimize_node = train_config.optimizer(train_config.learning_rate).minimize(loss_sum)
    initializer = tf.global_variables_initializer()

    # data define
    # TODO 建立data class，把所有的数据输入的问题全部封装
    epoch = 4
    batch_count = 3
    actual_batch_size = 10
    actual_test_size = 30
    train_x = np.random.random_integers(0, 1, [batch_count, max_time_stamp, actual_batch_size, x_depth])
    train_t = np.random.random_integers(0, 1, [batch_count, max_time_stamp, actual_batch_size, t_depth])
    test_x = np.random.random_integers(0, 1, [max_time_stamp, actual_test_size, x_depth])
    test_t = np.random.random_integers(0, 1, [max_time_stamp, actual_test_size, t_depth])

    train_metric_list = list()
    test_metric_list = list()

    with tf.Session() as sess:
        tf.summary.FileWriter(train_config.train_save_path, sess.graph)
        tf.summary.FileWriter(train_config.test_save_path, sess.graph)
        sess.run(initializer)

        for i in range(0, epoch):
            for j in range(0, batch_count):
                sess.run([optimize_node], feed_dict={placeholder_x: train_x[j], placeholder_t: train_t[j]})
                c_pred, r_pred = sess.run([c_pred_list, r_pred_list],
                                          feed_dict={placeholder_x: train_x[j], placeholder_t: train_t[j]})
                # TODO 在解决prediction的问题之前，暂时不用summary
                # train_summary.add_summary(summary)
                metric_result = pm.performance_measure(c_pred, r_pred, train_x[j], train_t[j],
                                                       model_config.max_time_stamp, actual_batch_size,
                                                       model_config.input_x_depth, model_config.threshold)
                train_metric_list.append([i, j, metric_result])

            c_pred, r_pred = sess.run([c_pred_list, r_pred_list],
                                      feed_dict={placeholder_x: test_x, placeholder_t: test_t})
            metric_result = pm.performance_measure(c_pred, r_pred, test_x, test_t, model_config.max_time_stamp,
                                                   actual_test_size, model_config.input_x_depth,
                                                   model_config.threshold)
            test_metric_list.append([i, None, metric_result])
            # test_summary.add_summary(summary)

        pm.save_result(train_config.train_save_path, 'train_result.csv', train_metric_list)
        pm.save_result(train_config.test_save_path, 'test_result.csv', test_metric_list)


def main():
    training_configuration, model_configuration = configuration_set()
    training(training_configuration, model_configuration)


if __name__ == '__main__':
    main()
