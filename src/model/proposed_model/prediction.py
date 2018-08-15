# coding=utf-8
import tensorflow as tf

import attention_mechanism
import auc_eval
import intensity
import revised_rnn
import rnn_config as config


class AttentionMixLayer(object):
    def __init__(self, model_configuration, revise_rnn, attention):
        # General Model Parameters
        self.__c_r_ratio = model_configuration.c_r_ratio
        self.__input_x_depth = model_configuration.input_x_depth
        self.__input_t_depth = model_configuration.input_t_depth
        self.__max_time_stamp = model_configuration.max_time_stamp
        self.__num_hidden = model_configuration.num_hidden
        self.__init_map = model_configuration.init_map

        # component
        self.__rnn = revise_rnn
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
        mutual_intensity = kwargs['mutual_intensity']
        if input_x is None or input_t is None:
            raise ValueError('kwargs should contain key parameter input_x, input_t')

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

        with tf.variable_scope('pred_para', reuse=tf.AUTO_REUSE):
            c_weight = tf.get_variable(name='classification_weight', shape=[self.__num_hidden, self.__x_depth],
                                       initializer=self.__init_map["classification_weight"], dtype=tf.float64)
            c_bias = tf.get_variable(name='classification_bias', shape=[self.__x_depth],
                                     initializer=self.__init_map["classification_bias"], dtype=tf.float64)
            r_weight = tf.get_variable(name='regression_weight', shape=[self.__num_hidden, self.__t_depth],
                                       initializer=self.__init_map["regression_weight"], dtype=tf.float64)
            r_bias = tf.get_variable(name='regression_bias', shape=[1, ],
                                     initializer=self.__init_map["regression_bias"], dtype=tf.float64)

        with tf.name_scope('output'):
            # a list with length equal to time_stamp, each element has the size [batch_size, num_hidden]
            mix_hidden_state_list = tf.unstack(mix_hidden_state_list, axis=0)

            un_c_pred_list = []
            r_pred_list = []
            with tf.name_scope('c_output'):
                for state in mix_hidden_state_list:
                    un_c_pred = tf.matmul(state, c_weight) + c_bias
                    un_c_pred_list.append(un_c_pred)
                un_c_pred_list = tf.convert_to_tensor(un_c_pred_list)
            with tf.name_scope('r_output'):
                for state in mix_hidden_state_list:
                    r_pred = tf.matmul(state, r_weight) + r_bias
                    r_pred_list.append(r_pred)
                r_pred_list = tf.convert_to_tensor(r_pred_list)

        # pred next events based on the previous information, so we don't predict the last input.
        with tf.name_scope('discard_last'):
            time_stamp = un_c_pred_list.get_shape()[0].value
            un_c_pred_list = un_c_pred_list[0:time_stamp - 1, :, :]
            r_pred_list = r_pred_list[0:time_stamp - 1, :, :]
            c_label = input_x[1:time_stamp, :, :]
            r_label = input_t[1:time_stamp, :, :]

        with tf.name_scope('loss'):
            # for the requirement of data structure, we need to truncate data or pad zero. The output of padding
            # state is useless

            with tf.name_scope('value_clip'):
                meaningful_data = tf.reduce_max(c_label, axis=2, keepdims=True)
                r_pred_list = r_pred_list * meaningful_data
                un_c_pred_list = un_c_pred_list * meaningful_data

            with tf.name_scope('c_loss'):
                # we use the binary entropy loss function proposed in Large-scale Multi-label Text Classification -
                # Revisiting Neural Networks, arxiv.org/pdf/1312.5419
                c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=c_label, logits=un_c_pred_list))
            with tf.name_scope('r_loss'):
                r_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=r_label, predictions=r_pred_list))
                r_loss = tf.cast(r_loss, dtype=tf.float64)

        with tf.name_scope('normal_c'):
            c_pred_list = tf.sigmoid(un_c_pred_list)

        return c_loss, r_loss, c_pred_list, r_pred_list, c_label, r_label


def performance_summary(input_x, input_t, c_pred, r_pred, threshold):
    # performance metrics are obtained based on A Review on Multi-Label Learning Algorithms,
    #  Zhang et al, TKDE, 2014
    # size [time_stamp, batch_size, depth]
    """
    def __top_k_coverage(x, prediction, k):
        coverage_sum = 0
        _, indices = tf.nn.top_k(prediction, k=k)
        indices_shape = indices.shape
        s = x[indices_shape]
        single_pred = tf.ceil(prediction[indices])

        return coverage_sum/(k*len(x))
    """

    def __confusion_matrix(x, prediction, th):
        floor = tf.floor(prediction)
        ceil = tf.ceil(prediction)
        pred_label = tf.cast(tf.where(prediction > th, floor, ceil), tf.bool)
        true_label = tf.cast(x, tf.bool)

        tpn = tf.reduce_sum(tf.cast(tf.logical_and(pred_label, true_label), dtype=tf.float64))
        tnn = tf.reduce_sum(tf.cast(tf.logical_not(tf.logical_or(pred_label, true_label)), dtype=tf.float64))
        fpn = tf.reduce_sum(tf.cast(tf.logical_xor(tf.logical_or(pred_label, true_label), true_label),
                                    dtype=tf.float64))
        fnn = tf.reduce_sum(tf.cast(tf.logical_not(pred_label), dtype=tf.float64)) - tnn
        return tpn, tnn, fpn, fnn

    with tf.name_scope('performance'):
        with tf.name_scope('macro_auc'):
            label = tf.reshape(input_x, [1, -1])
            pred = tf.reshape(c_pred, [1, -1])
            macro_auc = auc_eval.auc(labels=label, predictions=pred)
            tf.summary.scalar('macro_auc', tf.reduce_mean(macro_auc))

        with tf.name_scope('micro_auc'):
            label = tf.unstack(input_x, axis=2)
            pred = tf.unstack(c_pred, axis=2)
            micro_auc = 0
            for i in range(0, len(label)):
                micro_auc += auc_eval.auc(labels=label[i], predictions=pred[i])
            tf.summary.scalar('micro_auc', micro_auc / len(label))

        with tf.name_scope('confusion_matrix'):
            tp, tn, fp, fn = __confusion_matrix(input_x, c_pred, threshold)
        with tf.name_scope('one_error/top_1'):
            pass
        with tf.name_scope('top_5_coverage'):
            pass
        with tf.name_scope('top_10_coverage'):
            pass
        with tf.name_scope('coverage'):
            pass
        with tf.name_scope('acc'):
            acc = (tp + tn) / (tp + tn + fp + fn)
            tf.summary.scalar('c_accuracy', acc)
        with tf.name_scope('specificity'):
            specificity = tn / (tn + fp)
            tf.summary.scalar('specificity', specificity)
        with tf.name_scope('precision'):
            precision = tp / (tp + fp)
            tf.summary.scalar('c_precision', precision)
        with tf.name_scope('recall'):
            recall = tp / (tp + fn)
            tf.summary.scalar('c_recall', recall)
        with tf.name_scope('f1'):
            f_1 = 2 * precision * recall / (precision + recall)
            tf.summary.scalar('f1', f_1)
        with tf.name_scope('time_dev'):
            time_dev = tf.reduce_mean(tf.abs(r_pred - input_t))
            tf.summary.scalar('abs_time_deviation', time_dev)


def unit_test():
    train_config, model_config = config.validate_configuration_set()
    batch_size = model_config.batch_size

    placeholder_x = tf.placeholder('float64', [model_config.max_time_stamp, batch_size, model_config.input_x_depth])
    placeholder_t = tf.placeholder('float64', [model_config.max_time_stamp, batch_size, model_config.input_t_depth])

    # component define
    revise_gru_rnn = revised_rnn.RevisedRNN(model_configuration=model_config)
    intensity_component = intensity.Intensity(model_config=model_config)
    mutual_intensity = intensity_component.mutual_intensity_placeholder
    decay_function = tf.placeholder('float64', [model_config.time_decay_size])
    attention_model = attention_mechanism.HawkesBasedAttentionLayer(model_configuration=model_config,
                                                                    decay_function_place_holder=decay_function,
                                                                    mutual_intensity_placeholder=mutual_intensity)
    attention_layer = AttentionMixLayer(model_configuration=model_config, revise_rnn=revise_gru_rnn,
                                        attention=attention_model)
    prediction_layer = PredictionLayer(model_configuration=model_config)

    # model construct
    mix_state_list = attention_layer(input_x=placeholder_x, input_t=placeholder_t, mutual_intensity=mutual_intensity)
    c_loss, r_loss, c_pred_list, r_pred_list, c_label, r_label = \
        prediction_layer(mix_hidden_state_list=mix_state_list, input_x=placeholder_x, input_t=placeholder_t)

    performance_summary(input_x=c_label, input_t=r_label, c_pred=c_pred_list,
                        r_pred=r_pred_list, threshold=model_config.threshold)


if __name__ == "__main__":
    unit_test()
