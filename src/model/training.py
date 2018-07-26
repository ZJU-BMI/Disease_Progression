# coding=utf-8
import tensorflow as tf

import attention_mechanism
import configuration as config
import intensity
import performance_metrics as pm
import prediction
import read_data
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


# TODO 根据实际数据情况重写这部分代码
def configuration_set():
    train_config = config.TestConfiguration.get_test_training_config()
    model_config = config.TestConfiguration.get_test_model_config()
    return train_config, model_config


def training(train_config, model_config):
    # input define
    max_time_stamp = model_config.max_time_stamp
    batch_size = train_config.batch_size
    x_depth = model_config.input_x_depth
    t_depth = model_config.input_t_depth

    with tf.name_scope('input'):
        placeholder_x = tf.placeholder('float64', [max_time_stamp, batch_size, x_depth])
        placeholder_t = tf.placeholder('float64', [max_time_stamp, batch_size, t_depth])

    model = ModelEvaluate(training_config=train_config, model_config=model_config)

    c_loss, r_loss, c_pred_list, r_pred_list, merged_summary = model(input_x=placeholder_x, input_t=placeholder_t)
    with tf.name_scope('loss_sum'):
        loss_sum = c_loss + model_config.c_r_ratio * r_loss

    optimize_node = train_config.optimizer(train_config.learning_rate).minimize(loss_sum)
    initializer = tf.global_variables_initializer()

    # data define
    epoch = 4
    batch_count = 3
    actual_batch_size = 10
    actual_test_size = 30
    data = read_data.LoadData(actual_batch_size, "", x_depth, t_depth, max_time_stamp)

    train_metric_list = list()
    test_metric_list = list()

    with tf.Session() as sess:
        tf.summary.FileWriter(train_config.train_save_path, sess.graph)
        tf.summary.FileWriter(train_config.test_save_path, sess.graph)
        sess.run(initializer)

        for i in range(0, epoch):
            for j in range(0, batch_count):
                train_x, train_t = data.get_train_next_batch()
                sess.run([optimize_node], feed_dict={placeholder_x: train_x, placeholder_t: train_t})
                c_pred, r_pred = sess.run([c_pred_list, r_pred_list],
                                          feed_dict={placeholder_x: train_x, placeholder_t: train_t})
                # TODO 在解决prediction的问题之前，暂时不用summary
                # train_summary.add_summary(summary)
                metric_result = pm.performance_measure(c_pred, r_pred, train_x[1:max_time_stamp],
                                                       train_t[1:max_time_stamp], max_time_stamp - 1, actual_batch_size)
                train_metric_list.append([i, j, metric_result])

            test_x, test_t = data.get_test_data()
            c_pred, r_pred = sess.run([c_pred_list, r_pred_list],
                                      feed_dict={placeholder_x: test_x, placeholder_t: test_t})
            metric_result = pm.performance_measure(c_pred, r_pred, test_x[1:max_time_stamp], test_t[1:max_time_stamp],
                                                   max_time_stamp - 1, actual_test_size)
            test_metric_list.append([i, None, metric_result])
            # test_summary.add_summary(summary)

        pm.save_result(train_config.train_save_path, 'train_result.csv', train_metric_list)
        pm.save_result(train_config.test_save_path, 'test_result.csv', test_metric_list)


def main():
    training_configuration, model_configuration = configuration_set()
    training(training_configuration, model_configuration)


if __name__ == '__main__':
    main()
