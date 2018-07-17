import tensorflow as tf

import attention_mechanism
import rnn


# TODO 还需要完善文档，特别是对输入的参数的格式设定，要全部特别说明
class ModelConfiguration(object):
    def __init__(self, learning_rate, max_train_steps, batch_size, x_depth, t_depth, time_steps, num_hidden,
                 cell_type, summary_save_path, c_r_ratio, activation, zero_sate, init_strategy, mutual_intensity_path,
                 base_intensity_path, ):
        # Tensorboard Data And Output Save Path
        self.model_summary_save_path = summary_save_path

        # Training Parameters
        self.learning_rate = learning_rate
        self.max_training_steps = max_train_steps
        self.batch_size = batch_size

        # Model Parameters
        self.c_r_ratio = c_r_ratio
        self.input_x_depth = x_depth
        self.input_t_depth = t_depth
        self.time_steps = time_steps

        # Network Parameters
        self.num_hidden = num_hidden
        self.cell_type = cell_type
        self.activation = activation
        self.zero_state = zero_sate
        self.init_strategy = init_strategy

        # Attention Parameters
        self.mutual_intensity_path = mutual_intensity_path
        self.base_intensity_path = base_intensity_path


class AttentionBasedModel(object):
    def __init__(self, model_config):
        # Tensorboard Data And Output Save Path
        self.model_summary_save_path = model_config.summary_save_path

        # Training Parameters
        self.learning_rate = model_config.learning_rate
        self.max_training_steps = model_config.max_train_steps
        self.batch_size = model_config.batch_size

        # General Model Parameters
        self.c_r_ratio = model_config.c_r_ratio
        self.input_x_depth = model_config.x_depth
        self.input_t_depth = model_config.t_depth
        self.time_steps = model_config.time_steps

        # Network Parameters
        self.num_hidden = model_config.num_hidden
        self.cell_type = model_config.cell_type
        self.activation = model_config.activation
        self.zero_state = model_config.zero_sate
        self.init_strategy = model_config.init_strategy

        # Attention Parameters
        self.mutual_intensity_path = model_config.mutual_intensity_path
        self.base_intensity_path = model_config.base_intensity_path

        # Output Parameters
        self.c_weight = None
        self.c_bias = None
        self.r_weight = None
        self.r_bias = None

    def build(self):
        revised_rnn = rnn.RevisedRNN(time_stamp=self.time_steps, batch_size=self.batch_size, x_depth=self.input_x_depth,
                                     t_depth=self.input_t_depth, hidden_state=self.num_hidden,
                                     init_strategy_map=self.init_strategy, activation=self.activation,
                                     zero_state=self.zero_state, name='revised_rnn')
        intensity_component = attention_mechanism.Intensity(time_stamp=self.time_steps, batch_size=self.batch_size,
                                                            x_depth=self.input_x_depth, t_depth=self.input_t_depth,
                                                            mutual_intensity_path=self.mutual_intensity_path,
                                                            base_intensity_path=self.base_intensity_path,
                                                            name='intensity')
        attention_component = attention_mechanism.AttentionMechanism(revised_rnn, intensity_component)
        self.c_weight, self.c_bias, self.r_weight, self.r_bias = self.__output_parameter()

        hidden_states_list = revised_rnn(input_x=revised_rnn.input_x, input_t=revised_rnn.input_t)

        output_mix_hidden_state = []
        with tf.name_scope('attention_mix'):
            for time_stamps in range(1, self.time_steps + 1):
                weight = attention_component(time_stamps)
                mix_state = tf.matmul(weight, tf.unstack(hidden_states_list, axis=0))
                output_mix_hidden_state.append(mix_state)

        with tf.name_scope('prediction'):
            for state in output_mix_hidden_state:
                # TODO 尚未完成
                pass

    def __output_parameter(self):
        # TODO 修改初始化策略
        with tf.variable_scope('rnn_output_para', reuse=tf.AUTO_REUSE):
            c_weight = tf.get_variable(name='classification_weight', shape=[self.num_hidden, self.input_x_depth],
                                       initializer=tf.random_normal_initializer())
            c_bias = tf.get_variable(name='classification_bias', shape=[self.input_x_depth],
                                     initializer=tf.zeros_initializer())
            r_weight = tf.get_variable(name='regression_weight', shape=[self.num_hidden, self.input_t_depth],
                                       initializer=tf.random_normal_initializer())
            r_bias = tf.get_variable(name='regression_bias', shape=[1, ],
                                     initializer=tf.zeros_initializer())
        return c_weight, c_bias, r_weight, r_bias


def run():
    pass
