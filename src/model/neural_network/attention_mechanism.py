# coding=utf-8
import tensorflow as tf

import intensity
import rnn_config as config


class HawkesBasedAttentionLayer(object):
    def __init__(self, model_configuration, mutual_intensity_placeholder, decay_function_place_holder):
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
        self.__t_depth = model_configuration.input_t_depth
        self.__name = 'hawkes_based_attention'
        self.__init_map = model_configuration.init_map
        self.__decay_size = model_configuration.time_decay_size
        self.__mutual_intensity = mutual_intensity_placeholder
        self.__time_decay = decay_function_place_holder
        self.__init_argument_validation()
        self.__attention_parameter()

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

        weight = self.__calc_weight(input_x, input_t, time_index, mutual_intensity)

        state = []
        with tf.name_scope('mix_' + str(time_index)):
            with tf.name_scope('mix'):
                for i in range(0, time_index + 1):
                    state.append(weight[i] * hidden_tensor[i])
            state = tf.convert_to_tensor(state, tf.float64)
            with tf.name_scope('average'):
                mix_state = tf.reduce_sum(state, axis=0)
        return mix_state

    def __calc_weight(self, input_x, input_t, time_index, mutual_intensity):
        """
        :param input_x:
        :param input_t:
        :param time_index: the first output has the time index equal to zero
        :param mutual_intensity:
        :return: a normalized hidden state weight with size [time_index+1, batch_size, 1].
        """
        time_decay_placeholder = self.__time_decay

        with tf.name_scope('data_unstack'):
            input_x_list = tf.unstack(input_x, axis=0)
            input_t_list = tf.unstack(input_t, axis=0)
            input_x_list = input_x_list[0: time_index + 1]
            input_t_list = input_t_list[0: time_index + 1]

        with tf.name_scope('unnormal_weight'):
            last_time = input_t_list[time_index][0]
            weight_list = []

            # calculate weight
            for i in range(0, time_index + 1):
                intensity_sum = 0
                for j in range(0, i + 1):
                    with tf.name_scope('time_calc'):
                        time_interval = tf.cast(last_time - input_t[j], dtype=tf.int64)
                        time_onehot = tf.one_hot(time_interval, self.__decay_size, dtype=tf.float64)
                    with tf.name_scope('decay_calc'):
                        time_decay = time_onehot * time_decay_placeholder
                        time_decay = tf.reduce_sum(time_decay, axis=2)
                    with tf.name_scope('weight_calc'):
                        x_t_j = input_x_list[j]
                        single_intensity = tf.matmul(x_t_j, mutual_intensity)
                        # TODO 此处是否要加入base intensity，怎么加，需要继续想，暂时先不加
                        single_intensity = tf.matmul(single_intensity, self.__mutual_parameter) * time_decay
                    intensity_sum += single_intensity
                weight_list.append(intensity_sum)
            unnormalized_weight = tf.convert_to_tensor(weight_list, dtype=tf.float64)

        with tf.name_scope('weight'):
            intensity_sum = tf.reduce_sum(unnormalized_weight, axis=0, keep_dims=True)
            weight = unnormalized_weight / intensity_sum

        return weight

    def __init_argument_validation(self):
        if not (self.__init_map.__contains__('mutual_intensity') and self.__init_map.__contains__('combine')):
            raise ValueError('init map should contain elements with name, mutual_intensity, combine')

    def __attention_parameter(self):
        size = self.__x_depth
        with tf.variable_scope('attention_para', reuse=tf.AUTO_REUSE):
            mutual = tf.get_variable(name='mutual', shape=[size, 1], dtype=tf.float64,
                                     initializer=self.__init_map['mutual_intensity'])
            self.__mutual_parameter = mutual


def unit_test():
    train_config, model_config = config.validate_configuration_set()

    batch_size = model_config.batch_size
    placeholder_x = tf.placeholder('float64', [model_config.max_time_stamp, batch_size, model_config.input_x_depth])
    placeholder_t = tf.placeholder('float64', [model_config.max_time_stamp, batch_size, model_config.input_t_depth])
    hidden_tensor = tf.placeholder('float64', [model_config.max_time_stamp, batch_size, model_config.num_hidden])

    intensity_obj = intensity.Intensity(model_config)
    mutual_placeholder = intensity_obj.mutual_intensity_placeholder
    time_decay = tf.placeholder('float64', [model_config.time_decay_size])
    hawkes_attention = HawkesBasedAttentionLayer(model_config, mutual_placeholder, time_decay)

    for time_stamp in range(0, model_config.max_time_stamp):
        mix_state = hawkes_attention(time_stamp, hidden_tensor, placeholder_x, placeholder_t, mutual_placeholder)
        print(mix_state)


if __name__ == "__main__":
    unit_test()
