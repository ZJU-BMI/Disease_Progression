import numpy as np
import tensorflow as tf

import rnn


class AttentionMechanism(object):
    def __init__(self, revised_rnn, intensity):
        """
        :param revised_rnn: should be an instance of RevisedRNN having output hidden states tensor with shape
        [time_stamps, batch_size, hidden_states], the first time stamp should be one
        :param intensity:
        """
        self.__rnn = revised_rnn
        self.__intensity = intensity

    # 每call一次，输出一次hidden state的混合， 可变参数集需要指明时间点 t，以输出[batch_size, hidden_states]
    def __call__(self, *args, **kwargs):
        if not kwargs.__contains__('time_stamp') or (not isinstance(kwargs['time_stamp'], int)):
            raise ValueError('must contains parameter "time_stamp", and it must be int')
        time_stamp = kwargs['time_stamp']
        weight = self.weight_calc(time_stamp=time_stamp)
        return weight

    def argument_validation(self):
        if not isinstance(self.__rnn, rnn.RevisedRNN):
            raise ValueError('argument rnn must be an instance of RevisedRNN')
        if not isinstance(self.__intensity, Intensity):
            raise ValueError('argument mutual_intensity must be an instance of MutualIntensity')

    def weight_calc(self, time_stamp):
        weight = self.__intensity(time_stamp=time_stamp)
        return weight


class Intensity(object):
    def __init__(self, time_stamp, batch_size, x_depth, t_depth, mutual_intensity_path, base_intensity_path, name,
                 placeholder_x, placeholder_t):
        self.__x_depth = x_depth
        self.__t_depth = t_depth
        self.__time_stamp = time_stamp
        self.__batch_size = batch_size
        self.__name = name

        # get_intensity
        self.__mutual_intensity_path = mutual_intensity_path
        self.__base_intensity_path = base_intensity_path
        self.__base_intensity = self.read_base_intensity()
        self.__mutual_intensity = self.read_base_intensity()

        # define placeholder
        self.input_x = placeholder_x
        self.input_t = placeholder_t

        print('initialize rnn and build mutual intensity component accomplished')

    def read_mutual_intensity(self):
        sum_intensity = len(self.__mutual_intensity_path)
        return sum_intensity

    def read_base_intensity(self):
        sum_intensity = len(self.__base_intensity_path)
        return sum_intensity

    # 每call一次，均根据此时的时间信息，计算mutual intensity的调和平均数，需要使用时间信息，计算之前的信息混合
    def __call__(self, *args, **kwargs):
        if not kwargs.__contains__('time_stamp') or not isinstance(kwargs['time_stamp'], int) or kwargs['time_stamp'] \
                > self.input_x.shape[0].value:
            raise ValueError('kwargs must contains time_stamp, or value of time_stamp illegal')
        with tf.name_scope('data_unstack'):
            time_stamp = kwargs['time_stamp']
            input_x_list = tf.unstack(self.input_x, axis=0)
            input_t_list = tf.unstack(self.input_t, axis=0)
            input_x_list = input_x_list[0: time_stamp]
            input_t_list = input_t_list[0: time_stamp]

        with tf.name_scope('unnormal'):
            unnormalized_intensity = tf.convert_to_tensor(
                self.calculate_intensity(input_x_list, input_t_list, time_stamp), tf.float64)

        with tf.name_scope('weight'):
            intensity_sum = tf.reduce_sum(unnormalized_intensity, axis=0)
            weight = unnormalized_intensity / intensity_sum

        return weight

    # TODO 计算每一个值
    def calculate_intensity(self, input_x_list, input_t_list, time_stamp):
        node = tf.convert_to_tensor(np.ones([time_stamp], ), dtype=tf.float64)
        return node


def main():
    # AttentionMechanism()
    pass


if __name__ == "__main__":
    main()
