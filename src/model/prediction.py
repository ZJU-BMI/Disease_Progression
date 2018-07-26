# coding=utf-8
import tensorflow as tf

import attention_mechanism
import configuration as config
import intensity
import revised_rnn


class AttentionMixLayer(object):
    def __init__(self, model_configuration, revise_rnn, mutual_intensity, attention):
        # General Model Parameters
        self.__c_r_ratio = model_configuration.c_r_ratio
        self.__input_x_depth = model_configuration.input_x_depth
        self.__input_t_depth = model_configuration.input_t_depth
        self.__max_time_stamp = model_configuration.max_time_stamp
        self.__num_hidden = model_configuration.num_hidden
        self.__init_map = model_configuration.init_map

        # component
        self.__rnn = revise_rnn
        self.__mutual_intensity = mutual_intensity
        self.__attention = attention

    def __call__(self, **kwargs):
        """
        :param kwargs:
        kwargs['input_x']: a tensor with shape: [time_stamp, batch_size, x_depth]
        kwargs['input_t']: a tensor with shape: [time_stamp, batch_size, t_depth]
        :return:
        mix_hidden_state_list: a tensor with shape [time_stamp. batch_size, num_hidden]
        each state is the mix state under the process of attention mechanism
        """
        input_x = kwargs['input_x']
        input_t = kwargs['input_t']
        if input_x is None or input_t is None:
            raise ValueError('kwargs should contain key parameter input_x, input_t')

        mutual_intensity = self.__mutual_intensity
        hidden_tensor = self.__rnn(input_x, input_t)

        mix_hidden_state_list = []
        with tf.name_scope('attention'):
            for time_stamp in range(0, self.__max_time_stamp):
                with tf.name_scope('mix_state'):
                    mix_state = self.__attention(time_stamp, hidden_tensor, input_x, input_t, mutual_intensity)
                    mix_hidden_state_list.append(mix_state)
            mix_hidden_state_list = tf.convert_to_tensor(mix_hidden_state_list, dtype=tf.float64,
                                                         name='attention_states')
        return mix_hidden_state_list


class PredictionLayer(object):
    def __init__(self, model_configuration):
        self.__init_map = model_configuration.init_map
        self.__num_hidden = model_configuration.num_hidden
        self.__t_depth = model_configuration.input_t_depth
        self.__x_depth = model_configuration.input_x_depth
        self.__c_weight, self.__c_bias, self.__r_weight, self.__r_bias = self.__output_parameter()

    def __output_parameter(self):
        with tf.variable_scope('pred_para', reuse=tf.AUTO_REUSE):
            c_weight = tf.get_variable(name='classification_weight', shape=[self.__num_hidden, self.__x_depth],
                                       initializer=self.__init_map["classification_weight"], dtype=tf.float64)
            c_bias = tf.get_variable(name='classification_bias', shape=[self.__x_depth],
                                     initializer=self.__init_map["classification_bias"], dtype=tf.float64)
            r_weight = tf.get_variable(name='regression_weight', shape=[self.__num_hidden, self.__t_depth],
                                       initializer=self.__init_map["regression_weight"], dtype=tf.float64)
            r_bias = tf.get_variable(name='regression_bias', shape=[1, ],
                                     initializer=self.__init_map["regression_bias"], dtype=tf.float64)
        return c_weight, c_bias, r_weight, r_bias

    def __call__(self, **kwargs):
        """
        calculate the regression and classification loss, and return the prediction
        :param kwargs:
        :return:
        c_loss, classification_loss node
        r_loss, regression_loss node
        c_pred_list, unnormalized classification prediction with size [time_stamp-1, batch_size, x_depth]
        r_pred_list, regression prediction with size [time_stamp-1, batch_size, t_depth]
        """
        mix_hidden_state_list = kwargs['mix_hidden_state_list']
        input_x = kwargs['input_x']
        input_t = kwargs['input_t']

        if input_x is None or input_t is None or mix_hidden_state_list is None:
            raise ValueError('kwargs should contain key parameter input_x, input_t, and mix_hidden_state_list')

        with tf.name_scope('output'):
            # a list with length equal to time_stamp, each element has the size [batch_size, num_hidden]
            mix_hidden_state_list = tf.unstack(mix_hidden_state_list, axis=0)
            un_c_pred_list = []
            r_pred_list = []
            with tf.name_scope('c_output'):
                for state in mix_hidden_state_list:
                    un_c_pred = tf.matmul(state, self.__c_weight) + self.__c_bias
                    un_c_pred_list.append(un_c_pred)
                un_c_pred_list = tf.convert_to_tensor(un_c_pred_list)
                self.c_pred_node = un_c_pred_list
            with tf.name_scope('r_output'):
                for state in mix_hidden_state_list:
                    r_pred = tf.matmul(state, self.__r_weight) + self.__r_bias
                    r_pred_list.append(r_pred)
                r_pred_list = tf.convert_to_tensor(r_pred_list)
                self.r_pred_node = r_pred_list

            # pred next event based on the previous information
            with tf.name_scope('discard'):
                time_stamp = un_c_pred_list.get_shape()[0].value

                un_c_pred_list = un_c_pred_list[0:time_stamp - 1, :, :]
                r_pred_list = r_pred_list[0:time_stamp - 1, :, :]
                c_label = input_x[1:time_stamp, :, :]
                r_label = input_t[1:time_stamp, :, :]

        with tf.name_scope('loss'):
            with tf.name_scope('c_loss'):
                # we use the binary entropy loss function proposed in Large-scale Multi-label Text Classification -
                # Revisiting Neural Networks, arxiv.org/pdf/1312.5419
                c_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=c_label,
                                                                               logits=un_c_pred_list))
            with tf.name_scope('r_loss'):
                r_loss = tf.reduce_sum(tf.cast(tf.losses.mean_squared_error(labels=r_label, predictions=r_pred_list),
                                               dtype=tf.float64))

        return c_loss, r_loss, un_c_pred_list, r_pred_list


# TODO 在batch_size不定时无法使用，待解决
def performance_summary(input_x, input_t, c_pred, r_pred, threshold):
    # performance metrics are obtained based on A Review on Multi-Label Learning Algorithms,
    # Zhang et al, TKDE, 2014
    c_label = tf.cast(input_x, dtype=tf.bool)
    r_label = input_t

    c_auxiliary_one = tf.ones(c_pred.shape)
    c_auxiliary_zero = tf.zeros(c_pred.shape)
    c_pred_label = tf.where(c_pred > threshold, c_auxiliary_one, c_auxiliary_zero)
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


def unit_test():
    model_config = config.TestConfiguration.get_test_model_config()
    train_config = config.TestConfiguration.get_test_training_config()

    batch_size = train_config.batch_size

    placeholder_x = tf.placeholder('float64', [model_config.max_time_stamp, batch_size, model_config.input_x_depth])
    placeholder_t = tf.placeholder('float64', [model_config.max_time_stamp, batch_size, model_config.input_t_depth])

    # component define
    revise_gru_rnn = revised_rnn.RevisedRNN(model_configuration=model_config)
    intensity_component = intensity.Intensity(model_configuration=model_config)
    mutual_intensity = intensity_component.mutual_intensity
    attention_model = attention_mechanism.HawkesBasedAttentionLayer(model_configuration=model_config)
    attention_layer = AttentionMixLayer(model_configuration=model_config, mutual_intensity=mutual_intensity,
                                        revise_rnn=revise_gru_rnn, attention=attention_model)
    prediction_layer = PredictionLayer(model_configuration=model_config)

    # model construct
    mix_state_list = attention_layer(input_x=placeholder_x, input_t=placeholder_t)
    c_loss, r_loss, c_pred_list, r_pred_list = prediction_layer(mix_hidden_state_list=mix_state_list,
                                                                input_x=placeholder_x, input_t=placeholder_t)
    # performance_summary(input_x=placeholder_x, input_t=placeholder_t, c_pred=c_pred_list, r_pred=r_pred_list,
    #                     threshold=model_config.threshold)


if __name__ == "__main__":
    unit_test()
