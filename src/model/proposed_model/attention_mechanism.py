# coding=utf-8
import tensorflow as tf

import intensity
import rnn_config as config


class HawkesBasedAttentionLayer(object):
    def __init__(self, model_configuration, mutual_intensity_placeholder):
        """
        :param model_configuration contains
        x_depth:
        t_depth:
        name:
        init_map: should contain initializer with key mutual_intensity, combine
        time_decay_function: should be a list with size [1, 10000], each entry indicates the intensity at
        corresponding(the entry's index) day
        """
        self.__x_depth = model_configuration.input_x_depth
        self.__name = 'hawkes_based_attention'
        self.__init_map = model_configuration.init_map
        self.__time_decay_function_size = model_configuration.time_decay_size
        self.__mutual_intensity_placeholder = mutual_intensity_placeholder
        self.__init_argument_validation()

    def __call__(self, time_index, hidden_tensor, input_x, input_t, mutual_intensity):
        """
        get the mixed hidden state under the process of attention mechanism

        :param time_index: the time index, the first hidden state (not zero state) will be defined as time_index=0
        :param hidden_tensor: a hidden state tensor with size, [time_stamp, batch_size, hidden_state]
        :param input_x: tensor with size [max_time_stamp, batch_size, x_depth]
        :param input_t: tensor with size [max_time_stamp, batch_size, t_depth]
        :param mutual_intensity: a tensor with size [x_depth(event count), x_depth]
        :return: a mix hidden state at predefined time_index
        """

        weight = self.__calc_weight(input_x, time_index, mutual_intensity)
        state = []
        with tf.name_scope('mix_' + str(time_index)):
            with tf.name_scope('mix'):
                for i in range(0, time_index + 1):
                    state.append(weight[i] * hidden_tensor[i])
            state = tf.convert_to_tensor(state, tf.float64)
            with tf.name_scope('average'):
                mix_state = tf.reduce_sum(state, axis=0)
        return mix_state

    def __calc_weight(self, input_x, time_index, mi_placeholder):
        """
        :param input_x:
        :param time_index: the first output has the time index equal to zero
        :param mi_placeholder:
        :return: a normalized hidden state weight with size [time_index+1, batch_size, 1].
        """

        with tf.name_scope('data_unstack'):
            input_x_list = tf.unstack(input_x, axis=0)
            input_x_list = input_x_list[0: time_index + 1]

        with tf.variable_scope('attention_para', reuse=tf.AUTO_REUSE):
            mutual = tf.get_variable(name='mutual', shape=[self.__x_depth, 1], dtype=tf.float64,
                                     initializer=self.__init_map['mutual_intensity'])

        with tf.name_scope('unnormal_weight'):
            weight_list = []

            # calculate weight
            for i in range(0, time_index + 1):
                with tf.name_scope('weight_calc'):
                    x_t_j = input_x_list[i]
                    single_intensity = tf.matmul(x_t_j, mi_placeholder)
                    single_intensity = tf.matmul(single_intensity, mutual)
                    # TODO 这一问题留待后期以合适的情况加入
                    # time_decay 暂时删除时间因素
                weight_list.append(single_intensity)
            unnormalized_weight = tf.convert_to_tensor(weight_list, dtype=tf.float64)

        with tf.name_scope('weight'):
            intensity_sum = tf.reduce_sum(unnormalized_weight, axis=0, keepdims=True)
            weight = unnormalized_weight / intensity_sum

        return weight

    def __init_argument_validation(self):
        if not (self.__init_map.__contains__('mutual_intensity') and self.__init_map.__contains__('combine')):
            raise ValueError('init map should contain elements with name, mutual_intensity, combine')


def unit_test():
    train_config, model_config = config.validate_configuration_set()

    batch_size = model_config.batch_size
    placeholder_x = tf.placeholder('float64', [model_config.max_time_stamp, batch_size, model_config.input_x_depth])
    placeholder_t = tf.placeholder('float64', [model_config.max_time_stamp, batch_size, model_config.input_t_depth])
    hidden_tensor = tf.placeholder('float64', [model_config.max_time_stamp, batch_size, model_config.num_hidden])

    intensity_obj = intensity.Intensity(model_config)
    mutual_placeholder = intensity_obj.mutual_intensity_placeholder
    hawkes_attention = HawkesBasedAttentionLayer(model_config, mutual_placeholder)

    for time_stamp in range(0, model_config.max_time_stamp):
        mix_state = hawkes_attention(time_stamp, hidden_tensor, placeholder_x, placeholder_t, mutual_placeholder)
        print(mix_state)


if __name__ == "__main__":
    unit_test()
