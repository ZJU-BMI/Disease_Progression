# coding=utf-8
import os

import numpy as np
import tensorflow as tf

import attention_mechanism
import configuration
import intensity
import prediction
import revised_rnn


class


def unit_test():
    root_path = os.path.abspath('..\\..')

    # model config
    num_hidden = 3
    x_depth = 6
    t_depth = 1
    max_time_stamp = 4
    cell_type = 'revised_gru'
    threshold = 0.5
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
    # time decay由于日期是离散的，每一日的强度直接采用硬编码的形式写入
    time_decay_function = np.random.normal(0, 1, [10000, ])

    model_configuration = \
        configuration.ModelConfiguration(x_depth=x_depth, t_depth=t_depth,
                                         max_time_stamp=max_time_stamp, num_hidden=num_hidden, cell_type=cell_type,
                                         c_r_ratio=c_r_ratio, activation=activation,
                                         init_strategy=init_map, zero_state=zero_state, mutual_intensity_path=mi_path,
                                         base_intensity_path=bi_path, file_encoding=file_encoding, init_map=init_map,
                                         time_decay_function=time_decay_function, threshold=threshold)

    # input define
    batch_size = 7
    placeholder_x = tf.placeholder('float64', [max_time_stamp, batch_size, x_depth])
    placeholder_t = tf.placeholder('float64', [max_time_stamp, batch_size, t_depth])

    # component define
    revise_gru_rnn = revised_rnn.RevisedRNN(model_configuration=model_configuration)
    intensity_component = intensity.Intensity(model_configuration=model_configuration)
    mutual_intensity = intensity_component.mutual_intensity
    attention_model = attention_mechanism.HawkesBasedAttentionLayer(model_configuration=model_configuration)
    attention_layer = prediction.AttentionMixLayer(model_configuration=model_configuration,
                                                   mutual_intensity=mutual_intensity,
                                                   revise_rnn=revise_gru_rnn, attention=attention_model)
    prediction_layer = prediction.PredictionLayer(model_configuration=model_configuration)

    # model construct
    mix_state_list = attention_layer(input_x=placeholder_x, input_t=placeholder_t)
    c_loss, r_loss, c_pred_list, r_pred_list = prediction_layer(mix_hidden_state_list=mix_state_list,
                                                                input_x=placeholder_x, input_t=placeholder_t)
    prediction.performance_summary(input_x=placeholder_x, input_t=placeholder_t, c_pred=c_pred_list, r_pred=r_pred_list,
                                   threshold=threshold)
