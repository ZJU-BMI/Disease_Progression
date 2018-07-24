# coding=utf-8
import tensorflow as tf


class HawkesBasedAttentionLayer(object):
    def __init__(self, x_depth, t_depth, name, init_map, time_decay_function):
        """
        :param x_depth:
        :param t_depth:
        :param name:
        :param init_map: should contain initializer with key mutual_intensity, combine
        :param time_decay_function: should be a list with length at l0000, each entry indicates the intensity at
        corresponding(the entry's index) day
        """
        self.__x_depth = x_depth
        self.__t_depth = t_depth
        self.__name = name
        self.__init_map = init_map
        self.__time_decay_function = tf.convert_to_tensor(time_decay_function, dtype=tf.float64)
        self.__mutual_parameter = None

        self.__init_argument_validation()
        self.__attention_parameter()

    def __call__(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        :return:
        """
        self.__call_argument_validation(kwargs)
        time_stamp = kwargs['time_stamp']
        hidden_tensor = kwargs['hidden_tensor']
        input_x = kwargs['input_x']
        input_t = kwargs['input_t']
        mutual_intensity = kwargs['mutual_intensity']
        # base_intensity = kwargs['base_intensity'] base intensity unused

        weight = self.__calc_weight(input_x, input_t, time_stamp, mutual_intensity)

        state = []
        with tf.name_scope('mix'):
            with tf.name_scope('mix'):
                for i in range(0, time_stamp + 1):
                    state.append(weight[i] * hidden_tensor[i])
            state_list = tf.convert_to_tensor(state_list, tf.float64)
            with tf.name_scope('average'):
                mix_state = tf.reduce_sum(state_list, axis=0)
                output_mix_hidden_state.append(mix_state)
        return 1

    def __calc_weight(self, input_x, input_t, time_stamp, mutual_intensity):
        with tf.name_scope('data_unstack'):
            input_x_list = tf.unstack(input_x, axis=0)
            input_t_list = tf.unstack(input_t, axis=0)
            input_x_list = input_x_list[0: time_stamp + 1]
            input_t_list = input_t_list[0: time_stamp + 1]

        with tf.name_scope('unnormal_weight'):
            time_decay_function = self.__time_decay_function
            weight_list = []
            last_time = input_t_list[time_stamp][0]
            for i in range(0, time_stamp + 1):
                with tf.name_scope('weight'):
                    intensity_sum = 0
                    for j in range(0, i + 1):
                        with tf.name_scope('time_calc'):
                            time_interval = last_time - input_t[j]
                            time_interval = tf.cast(time_interval, dtype=tf.int64)
                            time_interval = tf.one_hot(time_interval, time_decay_function.shape[0], dtype=tf.float64)

                        with tf.name_scope('decay_calc'):
                            time_decay = time_interval * time_decay_function
                            time_decay = tf.reduce_sum(time_decay, axis=2)

                        with tf.name_scope('weight_calc'):
                            x_t = input_x_list[j]
                            intensity = tf.matmul(x_t, mutual_intensity)
                            intensity = tf.matmul(intensity, self.__mutual_parameter) * time_decay
                        intensity_sum += intensity
                    weight_list.append(intensity_sum)
            unnormalized_weight = tf.convert_to_tensor(weight_list, dtype=tf.float64)

        with tf.name_scope('weight'):
            intensity_sum = tf.expand_dims(tf.reduce_sum(unnormalized_weight, axis=1), axis=2)
            weight = unnormalized_weight / intensity_sum

        return weight

    @staticmethod
    def __call_argument_validation(arg_map):
        if not arg_map.__contains__('time_stamp') or (not isinstance(arg_map['time_stamp'], int)):
            raise ValueError('must contains parameter "time_stamp", and it must be int')

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
    # AttentionMechanism()
    pass


if __name__ == "__main__":
    unit_test()
