import numpy as np
import tensorflow as tf

import rnn


# TODO,整个模型要花点时间重写所有的变量的可访问性
class AttentionMechanism(object):
    def __init__(self, revised_rnn, intensity):
        """
        :param revised_rnn: should be an instance of RevisedRNN having output hidden states tensor with shape
        [time_stamps, batch_size, hidden_states], the first time stamp should be one
        :param intensity:
        """
        self.rnn = revised_rnn
        self.intensity = intensity
        self.hidden_states = tf.unstack(revised_rnn.states_tensor, axis=0)

    # 每call一次，输出一次hidden state的混合， 可变参数集需要指明时间点 t，以输出[batch_size, hidden_states]
    def __call__(self, *args, **kwargs):
        if not kwargs.__contains__('time_stamp') or (not isinstance(kwargs['time_stamp'], int)):
            raise ValueError('must contains parameter "time_stamp", and it must be int')
        time_stamp = kwargs['time_stamp']
        valid_hidden_states = self.hidden_states[0: time_stamp]
        weight = self.weight_calc(time_stamp=time_stamp)

        mix_hidden_state = tf.reduce_sum(weight * valid_hidden_states, axis=0)
        return mix_hidden_state

    def argument_validation(self):
        if not isinstance(self.rnn, rnn.RevisedRNN):
            raise ValueError('argument rnn must be an instance of RevisedRNN')
        if not isinstance(self.intensity, Intensity):
            raise ValueError('argument mutual_intensity must be an instance of MutualIntensity')

    def weight_calc(self, time_stamp):
        weight = self.intensity(time_stamp=time_stamp)
        return weight


class Intensity(object):
    def __init__(self, time_stamp, batch_size, x_depth, t_depth, mutual_intensity_path, base_intensity_path, name):
        self.x_depth = x_depth
        self.t_depth = t_depth
        self.time_stamp = time_stamp
        self.batch_size = batch_size

        self.mutual_intensity_path = mutual_intensity_path
        self.base_intensity_path = base_intensity_path
        self.base_intensity = self.read_base_intensity()
        self.mutual_intensity = self.read_base_intensity()

        self.input_x = None
        self.input_t = None
        self.name = name

        self.build()
        print('initialize rnn and build mutual intensity component accomplished')

    # TODO,其实此处的输入和RNN是一样的，回头确认一下是不是可以通过一次输入完成模型构建
    def build(self):
        with tf.name_scope('attention_input'):
            self.input_x = tf.placeholder(dtype=tf.float64, shape=[self.time_stamp, self.batch_size, self.x_depth],
                                          name='input_x')
            self.input_t = tf.placeholder(dtype=tf.float64, shape=[self.time_stamp, self.batch_size, self.t_depth],
                                          name='input_t')

    def read_mutual_intensity(self):
        sum_intensity = len(self.mutual_intensity_path)
        return sum_intensity

    def read_base_intensity(self):
        sum_intensity = len(self.base_intensity_path)
        return sum_intensity

    # 每call一次，均根据此时的时间信息，计算mutual intensity的调和平均数，需要使用时间信息，计算之前的信息混合
    def __call__(self, *args, **kwargs):
        if not kwargs.__contains__('time_stamp') or not isinstance(kwargs['time_stamp'], int) or kwargs['time_stamp'] \
                > self.input_x.shape[0].value:
            raise ValueError('kwargs must contains time_stamp, or value of time_stamp illegal')
        time_stamp = kwargs['time_stamp']
        input_x_list = tf.unstack(self.input_x, axis=0)
        input_t_list = tf.unstack(self.input_t, axis=0)
        input_x_list = input_x_list[0: time_stamp]
        input_t_list = input_t_list[0: time_stamp]

        unnormalized_intensity = tf.convert_to_tensor(self.calculate_intensity(input_x_list, input_t_list), tf.float64)
        intensity_sum = tf.reduce_sum(unnormalized_intensity, axis=0)
        weight = unnormalized_intensity / intensity_sum
        return weight

    # TODO 计算每一个值
    def calculate_intensity(self, input_x_list, input_t_list):
        return tf.convert_to_tensor(np.ones([self.time_stamp], ), dtype=tf.float64)
