import csv
import os

import numpy as np
import sklearn.metrics as sk_metric
import tensorflow as tf

import attention_mechanism
import rnn


class ModelConfiguration(object):
    def __init__(self, learning_rate, batch_size, x_depth, time_stamps, num_hidden, cell_type, summary_save_path,
                 c_r_ratio, activation, init_strategy, mutual_intensity_path, base_intensity_path, zero_state,
                 file_encoding, time_decay_function, t_depth=1, threshold=0.5):
        """
        :param learning_rate: should be a scalar
        :param batch_size: the size of minibatch, a scalar
        :param x_depth: defines the dimension of the input_x in a specific time stamp, it also indicates the number
        of type of event
        :param t_depth: defines the time of a specific time stamp, raise error if it is not 1
        :param time_stamps: should be a scalar, the length of RNN
        :param num_hidden: should be a scalar, the dimension of a hidden state
        :param cell_type: should be a string, 'revised_gru' or 'gru'
        :param summary_save_path: a folder path, used to store information of tensorboard
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
        # Tensorboard Data And Output Save Path
        self.model_summary_save_path = summary_save_path

        # Training Parameters
        self.learning_rate = learning_rate

        self.time_decay_function = time_decay_function

        # Model Parameters
        self.c_r_ratio = c_r_ratio
        self.input_x_depth = x_depth
        self.input_t_depth = t_depth
        self.time_stamps = time_stamps

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

        self.threshold = threshold


class AttentionBasedModel(object):
    def __init__(self, model_config):
        # Tensorboard Data And Output Save Path
        self.__model_summary_save_path = model_config.model_summary_save_path

        # Training Parameters
        self.__learning_rate = model_config.learning_rate

        # General Model Parameters
        self.__c_r_ratio = model_config.c_r_ratio
        self.__input_x_depth = model_config.input_x_depth
        self.__input_t_depth = model_config.input_t_depth
        self.__time_stamps = model_config.time_stamps
        self.__threshold = model_config.threshold

        # Network Parameters
        self.__num_hidden = model_config.num_hidden
        self.__cell_type = model_config.cell_type
        self.__activation = model_config.activation
        self.__zero_state = model_config.zero_state
        self.__init_strategy = model_config.init_strategy

        # Attention Parameters
        self.__mutual_intensity_path = model_config.mutual_intensity_path
        self.__base_intensity_path = model_config.base_intensity_path
        self.__file_encoding = model_config.file_encoding

        self.__time_decay_function = model_config.time_decay_function

        # Output Parameters
        self.__c_weight = None
        self.__c_bias = None
        self.__r_weight = None
        self.__r_bias = None

        # data feed node
        self.input_data_x = None
        self.input_data_t = None

        # expose_node
        self.merge_summary = None
        self.c_pred_node = None
        self.r_pred_node = None
        self.optimize = None

        self.build()

    def build(self):
        with tf.name_scope('input_data'):
            self.input_data_x = tf.placeholder('float64', shape=[self.__time_stamps, None,
                                                                 self.__input_x_depth], name='input_x')
            self.input_data_t = tf.placeholder('float64', shape=[self.__time_stamps, None,
                                                                 self.__input_t_depth], name='input_t')

        revised_rnn = rnn.RevisedRNN(time_stamp=self.__time_stamps,
                                     x_depth=self.__input_x_depth, t_depth=self.__input_t_depth,
                                     hidden_state=self.__num_hidden, init_strategy_map=self.__init_strategy,
                                     activation=self.__activation, zero_state=self.__zero_state,
                                     input_x=self.input_data_x, input_t=self.input_data_t)
        intensity_component = attention_mechanism.Intensity(time_stamp=self.__time_stamps, batch_size=None,
                                                            x_depth=self.__input_x_depth, t_depth=self.__input_t_depth,
                                                            mutual_intensity_path=self.__mutual_intensity_path,
                                                            base_intensity_path=self.__base_intensity_path,
                                                            name='intensity', placeholder_x=self.input_data_x,
                                                            placeholder_t=self.input_data_t,
                                                            file_encoding=self.__file_encoding,
                                                            para_init_map=self.__init_strategy,
                                                            time_decay_function=self.__time_decay_function)

        attention_component = attention_mechanism.AttentionMechanism(revised_rnn, intensity_component)
        self.__c_weight, self.__c_bias, self.__r_weight, self.__r_bias = self.__output_parameter()

        hidden_states_list = revised_rnn.states_tensor

        output_mix_hidden_state = []
        with tf.name_scope('attention'):
            for time_stamp in range(0, self.__time_stamps):
                # we allow time_stamp can be zero because zero-state has output
                with tf.name_scope('mix_state'):
                    with tf.name_scope('weight'):
                        weight = attention_component(time_stamp=time_stamp)
                    with tf.name_scope('states'):
                        hidden_states = tf.unstack(hidden_states_list)[0: time_stamp + 1]
                    state_list = []
                    with tf.name_scope('mix'):
                        with tf.name_scope('mix'):
                            for i in range(0, time_stamp + 1):
                                state_list.append(weight[i] * hidden_states[i])
                        state_list = tf.convert_to_tensor(state_list, tf.float64)
                        with tf.name_scope('average'):
                            mix_state = tf.reduce_sum(state_list, axis=0)
                            output_mix_hidden_state.append(mix_state)
            output_mix_hidden_state = tf.convert_to_tensor(output_mix_hidden_state, dtype=tf.float64,
                                                           name='attention_states')

        with tf.name_scope('output'):
            output_mix_hidden_state = tf.unstack(output_mix_hidden_state, axis=0)
            c_pred_list = []
            r_pred_list = []
            with tf.name_scope('c_output'):
                for state in output_mix_hidden_state:
                    c_pred = tf.matmul(state, self.__c_weight) + self.__c_bias
                    c_pred_list.append(c_pred)
                c_pred_list = tf.convert_to_tensor(c_pred_list)
                self.c_pred_node = c_pred_list
            with tf.name_scope('r_output'):
                for state in output_mix_hidden_state:
                    r_pred = tf.matmul(state, self.__r_weight) + self.__r_bias
                    r_pred_list.append(r_pred)
                r_pred_list = tf.convert_to_tensor(r_pred_list)
                self.r_pred_node = r_pred_list

        with tf.name_scope('normal_pred'):
            self.__performance_measure(c_pred_list, r_pred_list)

        with tf.name_scope('loss'):
            with tf.name_scope('c_loss'):
                # we use the binary entropy loss function proposed in Large-scale Multi-label Text Classification -
                # Revisiting Neural Networks, arxiv.org/pdf/1312.5419
                c_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_data_x,
                                                                               logits=c_pred_list))
            with tf.name_scope('r_loss'):
                r_loss = tf.reduce_sum(tf.cast(tf.losses.mean_squared_error(labels=self.input_data_t,
                                                                            predictions=r_pred_list),
                                               dtype=tf.float64))
            with tf.name_scope('loss_sum'):
                loss_sum = c_loss + self.__c_r_ratio * r_loss

            with tf.name_scope('summary'):
                tf.summary.scalar('loss_c', c_loss)
                tf.summary.scalar('loss_r', r_loss)
                tf.summary.scalar('loss_sum', loss_sum)

        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer()
            self.optimize = optimizer.minimize(loss_sum)

        self.merge_summary = tf.summary.merge_all()

    def __output_parameter(self):
        with tf.variable_scope('pred_para', reuse=tf.AUTO_REUSE):
            c_weight = tf.get_variable(name='classification_weight', shape=[self.__num_hidden, self.__input_x_depth],
                                       initializer=self.__init_strategy["classification_weight"], dtype=tf.float64)
            c_bias = tf.get_variable(name='classification_bias', shape=[self.__input_x_depth],
                                     initializer=self.__init_strategy["classification_bias"], dtype=tf.float64)
            r_weight = tf.get_variable(name='regression_weight', shape=[self.__num_hidden, self.__input_t_depth],
                                       initializer=self.__init_strategy["regression_weight"], dtype=tf.float64)
            r_bias = tf.get_variable(name='regression_bias', shape=[1, ],
                                     initializer=self.__init_strategy["regression_bias"], dtype=tf.float64)
        return c_weight, c_bias, r_weight, r_bias

    def __performance_measure(self, c_pred, r_pred):
        # performance metrics are obtained based on A Review on Multi-Label Learning Algorithms,
        # Zhang et al, TKDE, 2014
        c_label = tf.cast(self.input_data_x, dtype=tf.bool)
        r_label = self.input_data_t

        c_auxiliary_one = tf.cast(tf.ones(c_pred.shape, dtype=tf.int8), dtype=tf.bool)
        c_auxiliary_zero = tf.cast(tf.zeros(c_pred.shape, dtype=tf.int8), dtype=tf.bool)
        c_pred_label = tf.where(c_pred > self.__threshold, c_auxiliary_one, c_auxiliary_zero)
        with tf.name_scope('acc'):
            acc = tf.reduce_sum(tf.cast(tf.logical_and(c_pred_label, c_label), dtype=tf.float32)) / \
                  tf.reduce_sum(tf.cast(tf.logical_or(c_pred_label, c_label), dtype=tf.float32))
            tf.summary.scalar('c_accuracy', acc)
        with tf.name_scope('precision'):
            precision = tf.reduce_sum(tf.cast(tf.logical_and(c_pred_label, c_label), dtype=tf.float32)) / \
                        tf.reduce_sum(tf.cast(c_pred_label, dtype=tf.float32))
            tf.summary.scalar('c_precision', precision)
        with tf.name_scope('recall'):
            recall = tf.reduce_sum(tf.cast(tf.logical_and(c_pred_label, c_label), dtype=tf.float32)) / \
                     tf.reduce_sum(tf.cast(c_label, dtype=tf.float32))
            tf.summary.scalar('c_recall', recall)
        with tf.name_scope('f1'):
            f_1 = precision * recall / (precision + recall)
            tf.summary.scalar('c_f_1', f_1)
        with tf.name_scope('hloss'):
            denominator = (c_label.shape[0] * c_label.shape[1] * c_label.shape[2]).value
            difference = tf.cast(tf.logical_xor(c_label, c_pred_label), dtype=tf.float32)
            hamming_loss = tf.reduce_sum(difference) / denominator
            tf.summary.scalar('hamming_loss', hamming_loss)

        with tf.name_scope('time_dev'):
            time_dev = tf.reduce_mean(tf.abs(r_pred - r_label))
            tf.summary.scalar('abs_time_deviation', time_dev)


def performance_measure(c_pred, r_pred, c_label, r_label, time_stamp, batch_size, input_depth, threshold):
    # performance metrics are obtained based on A Review on Multi-Label Learning Algorithms,
    # Zhang et al, TKDE, 2014
    c_auxiliary_one = np.ones(c_pred.shape)
    c_auxiliary_zero = np.zeros(c_pred.shape)
    c_pred_label = np.where(c_pred > threshold, c_auxiliary_one, c_auxiliary_zero)

    acc = np.sum(np.logical_and(c_pred_label, c_label)) / np.sum(np.logical_or(c_pred_label, c_label))
    precision = np.sum(np.logical_and(c_pred_label, c_label)) / np.sum(c_pred_label)
    recall = np.sum(np.logical_and(c_pred_label, c_label)) / np.sum(c_label)
    f_1 = precision * recall / (precision + recall)

    # hamming loss
    denominator = c_label.shape[0] * c_label.shape[1] * c_label.shape[2]
    difference = np.logical_xor(c_pred_label, c_label)
    hamming_loss = np.sum(difference) / denominator

    c_label = np.reshape(c_label, [time_stamp * batch_size, input_depth])
    c_pred_label = np.reshape(c_pred_label, [time_stamp * batch_size, input_depth])
    coverage = sk_metric.coverage_error(c_label, c_pred_label)
    rank_loss = sk_metric.label_ranking_loss(c_label, c_pred_label)
    average_precision = sk_metric.average_precision_score(c_label, c_pred_label)
    macro_auc = sk_metric.roc_auc_score(c_label, c_pred_label, average='macro')
    micro_auc = sk_metric.roc_auc_score(c_label, c_pred_label, average='micro')
    time_dev = np.sum(np.abs(r_pred - r_label))

    metrics_map = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f_1, 'hamming_loss': hamming_loss,
                   'coverage': coverage, 'ranking_loss': rank_loss, 'average_precision': average_precision,
                   'macro_auc': macro_auc, 'micro_auc': micro_auc, 'absolute_time_deviation': time_dev}
    return metrics_map


def save_result(path, file_name, data):
    matrix_to_write = []
    head = ['epoch', 'batch', 'acc', 'precision', 'recall', 'f1', 'hamming_loss', 'coverage', 'ranking_loss',
            'average_precision', 'macro_auc', 'micro_auc', 'absolute_time_deviation']
    matrix_to_write.append(head)
    for item in data:
        epoch = item[0]
        batch = item[1]
        acc = item[2]['acc']
        precision = item[2]['precision']
        recall = item[2]['recall']
        f1 = item[2]['f1']
        hamming_loss = item[2]['hamming_loss']
        coverage = item[2]['coverage']
        ranking_loss = item[2]['ranking_loss']
        average_precision = item[2]['average_precision']
        macro_auc = item[2]['macro_auc']
        micro_auc = item[2]['micro_auc']
        absolute_time_deviation = item[2]['absolute_time_deviation']
        single_result = [epoch, batch, acc, precision, recall, f1, hamming_loss, coverage, ranking_loss,
                         average_precision, macro_auc, micro_auc, absolute_time_deviation]
        matrix_to_write.append(single_result)

    with open(path + file_name, 'w', encoding='utf-8-sig', newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(matrix_to_write)


def main():
    root_path = os.path.abspath('..\\..')

    save_path = root_path + "\\src\\model\\train"
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

    num_hidden = 3
    batch_size = 8
    x_depth = 6
    t_depth = 1
    time_stamps = 4
    cell_type = 'revised_gru'
    zero_state = np.random.normal(0, 1, [num_hidden, ])

    mi_path = root_path + "\\resource\\mutual_intensity_sample.csv"
    bi_path = root_path + "\\resource\\base_intensity_sample.csv"
    file_encoding = 'utf-8-sig'
    c_r_ratio = 1
    threshold = 0.01
    learning_rate = 10

    epoch = 100
    train_batch_count = 5
    test_batch_count = 5

    print(save_path)

    # time decay由于日期是离散的，每一日的强度直接采用硬编码的形式写入
    time_decay_function = np.random.normal(0, 1, [10000, ])

    model_config = ModelConfiguration(learning_rate=learning_rate, batch_size=batch_size, x_depth=x_depth,
                                      t_depth=t_depth, time_stamps=time_stamps, num_hidden=num_hidden,
                                      cell_type=cell_type, summary_save_path=save_path, c_r_ratio=c_r_ratio,
                                      activation=activation, init_strategy=init_map, zero_state=zero_state,
                                      mutual_intensity_path=mi_path, base_intensity_path=bi_path,
                                      file_encoding=file_encoding, time_decay_function=time_decay_function)

    attention_model = AttentionBasedModel(model_config)
    init = tf.global_variables_initializer()

    train_x = np.random.random_integers(0, 1, [train_batch_count, time_stamps, batch_size, x_depth])
    train_t = np.random.poisson(10, [train_batch_count, time_stamps, batch_size, t_depth])

    performance_list = []
    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(save_path, sess.graph)
        for i in range(0, epoch):
            for j in range(0, train_batch_count):
                input_x = train_x[j]
                input_t = train_t[j]
                merge, _ = sess.run([attention_model.merge_summary, attention_model.optimize],
                                    feed_dict={attention_model.input_data_x: input_x,
                                               attention_model.input_data_t: input_t})
                train_writer.add_summary(merge, j)

                c_pred, r_pred = sess.run([attention_model.c_pred_node, attention_model.r_pred_node],
                                          feed_dict={attention_model.input_data_x: input_x,
                                                     attention_model.input_data_t: input_t})
                metrics_map = performance_measure(c_pred, r_pred, input_x, input_t, time_stamps, batch_size,
                                                  x_depth, threshold)
                performance_list.append([i, j, metrics_map])
        save_result(save_path, '\\training_result.csv', performance_list)


if __name__ == "__main__":
    main()
