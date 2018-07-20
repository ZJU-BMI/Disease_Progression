import csv

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
                 placeholder_x, placeholder_t, file_encoding, para_init_map, time_decay_function):
        self.__x_depth = x_depth
        self.__t_depth = t_depth
        self.__time_stamp = time_stamp
        self.__batch_size = batch_size
        self.__name = name
        self.__file_encoding = file_encoding
        self.__para_init_map = para_init_map

        # get_intensity
        self.__mutual_intensity_path = mutual_intensity_path
        self.__base_intensity_path = base_intensity_path
        self.__base_intensity = self.__read_base_intensity()
        self.__mutual_intensity = self.__read_mutual_intensity()

        # define placeholder
        self.input_x = placeholder_x
        self.input_t = placeholder_t

        # define parameter
        self.__init_strategy = para_init_map
        self.__mutual_parameter = None

        self.__time_decay_function = tf.convert_to_tensor(time_decay_function, dtype=tf.float64)

        self.__attention_parameter()

        self.__argument_check()
        print('initialize rnn and build mutual intensity component accomplished')

    def __argument_check(self):
        if not (self.__init_strategy.__contains__('mutual_intensity') and self.__init_strategy.__contains__('combine')):
            raise ValueError('init map should contain elements with name, mutual_intensity, combine')

    def __attention_parameter(self):
        size = self.__x_depth
        with tf.variable_scope('att_para', reuse=tf.AUTO_REUSE):
            mutual = tf.get_variable(name='mutual', shape=[size, 1], dtype=tf.float64,
                                     initializer=self.__init_strategy['mutual_intensity'])
            self.__mutual_parameter = mutual

    def __read_mutual_intensity(self):
        mutual_file_path = self.__mutual_intensity_path
        encoding = self.__file_encoding
        size = self.__x_depth

        mutual_intensity = np.zeros([size, size])
        with open(mutual_file_path, 'r', encoding=encoding, newline="") as file:
            csv_reader = csv.reader(file)
            row_index = 0
            for line in csv_reader:
                if len(line) != size:
                    raise ValueError('mutual intensity incompatible')
                for col_index in range(0, size):
                    mutual_intensity[row_index][col_index] = line[col_index]
                row_index += 1
            if row_index != size:
                raise ValueError('mutual intensity incompatible')

        with tf.name_scope('m_intensity'):
            mutual_intensity = tf.convert_to_tensor(mutual_intensity, dtype=tf.float64)

        return mutual_intensity

    def __read_base_intensity(self):
        base_file_path = self.__base_intensity_path
        encoding = self.__file_encoding
        size = self.__x_depth

        base_intensity = np.zeros([1, size])
        with open(base_file_path, 'r', encoding=encoding, newline="") as file:
            csv_reader = csv.reader(file)
            for line in csv_reader:
                if len(line) != size:
                    raise ValueError('mutual intensity incompatible')
                for col_index in range(0, size):
                    base_intensity[0][col_index] = line[col_index]
                break

        with tf.name_scope('b_intensity'):
            base_intensity = tf.convert_to_tensor(base_intensity, dtype=tf.float64)
        return base_intensity

    # 每call一次，均根据此时的时间信息，计算mutual intensity的调和平均数，需要使用时间信息，计算之前的信息混合
    def __call__(self, *args, **kwargs):
        if not kwargs.__contains__('time_stamp') or not isinstance(kwargs['time_stamp'], int) or kwargs['time_stamp'] \
                > self.input_x.shape[0].value:
            raise ValueError('kwargs must contains time_stamp, or value of time_stamp illegal')

        with tf.name_scope('data_unstack'):
            time_stamp = kwargs['time_stamp']
            input_x_list = tf.unstack(self.input_x, axis=0)
            input_t_list = tf.unstack(self.input_t, axis=0)
            input_x_list = input_x_list[0: time_stamp + 1]
            input_t_list = input_t_list[0: time_stamp + 1]

        with tf.name_scope('unnormal'):
            unnormalized_intensity = tf.convert_to_tensor(
                self.__calculate_intensity(input_x_list, input_t_list, time_stamp), tf.float64)

        with tf.name_scope('weight'):
            intensity_sum = tf.expand_dims(tf.reduce_sum(unnormalized_intensity, axis=1), axis=2)
            weight = unnormalized_intensity / intensity_sum

        return weight

    def __calculate_intensity(self, input_x, input_t, time_stamp):

        time_decay_function = self.__time_decay_function
        weight_list = []

        mutual = self.__mutual_intensity
        last_time = input_t[time_stamp][0]
        for i in range(0, time_stamp + 1):
            with tf.name_scope('weight'):
                intensity_sum = 0
                for j in range(0, i + 1):
                    with tf.name_scope('time_calc'):
                        time_interval = last_time - self.input_t[j]
                        time_interval = tf.cast(time_interval, dtype=tf.int64)
                        time_interval = tf.one_hot(time_interval, time_decay_function.shape[0], dtype=tf.float64)

                    with tf.name_scope('decay_calc'):
                        time_decay = time_interval * time_decay_function
                        time_decay = tf.reduce_sum(time_decay, axis=2)

                    with tf.name_scope('weight_calc'):
                        x_t = input_x[j]
                        intensity = tf.matmul(x_t, mutual)
                        intensity = tf.matmul(intensity, self.__mutual_parameter) * time_decay
                    intensity_sum += intensity
                weight_list.append(intensity_sum)
        weight_list = tf.convert_to_tensor(weight_list, dtype=tf.float64)
        return weight_list


def main():
    # AttentionMechanism()
    pass


if __name__ == "__main__":
    main()
