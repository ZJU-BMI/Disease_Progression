import numpy as np
import tensorflow as tf

import attention_mechanism
import rnn


# TODO 还需要完善文档，特别是对输入的参数的格式设定，要全部特别说明
class ModelConfiguration(object):
    def __init__(self, learning_rate, max_train_steps, batch_size, x_depth, t_depth, time_stamps, num_hidden,
                 cell_type, summary_save_path, c_r_ratio, activation, init_strategy, mutual_intensity_path,
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
        self.time_stamps = time_stamps

        # Network Parameters
        self.num_hidden = num_hidden
        self.cell_type = cell_type
        self.activation = activation
        # TODO 初始化策略
        self.zero_state = np.random.normal(0, 1, [num_hidden, ])
        self.init_strategy = init_strategy

        # Attention Parameters
        self.mutual_intensity_path = mutual_intensity_path
        self.base_intensity_path = base_intensity_path


class AttentionBasedModel(object):
    def __init__(self, model_config):
        # Tensorboard Data And Output Save Path
        self.model_summary_save_path = model_config.model_summary_save_path

        # Training Parameters
        self.learning_rate = model_config.learning_rate
        self.max_training_steps = model_config.max_training_steps
        self.batch_size = model_config.batch_size

        # General Model Parameters
        self.c_r_ratio = model_config.c_r_ratio
        self.input_x_depth = model_config.input_x_depth
        self.input_t_depth = model_config.input_t_depth
        self.time_stamps = model_config.time_stamps

        # Network Parameters
        self.num_hidden = model_config.num_hidden
        self.cell_type = model_config.cell_type
        self.activation = model_config.activation
        self.zero_state = model_config.zero_state
        self.init_strategy = model_config.init_strategy

        # Attention Parameters
        self.mutual_intensity_path = model_config.mutual_intensity_path
        self.base_intensity_path = model_config.base_intensity_path

        # Output Parameters
        self.c_weight = None
        self.c_bias = None
        self.r_weight = None
        self.r_bias = None

        self.input_data_x = None
        self.input_data_t = None

        self.build()

    def build(self):
        with tf.name_scope('input_data'):
            self.input_data_x = tf.placeholder('float64', shape=[self.time_stamps, self.batch_size,
                                                                 self.input_x_depth], name='input_x')
            self.input_data_t = tf.placeholder('float64', shape=[self.time_stamps, self.batch_size, self.input_t_depth],
                                               name='input_t')

        revised_rnn = rnn.RevisedRNN(time_stamp=self.time_stamps, batch_size=self.batch_size,
                                     x_depth=self.input_x_depth, t_depth=self.input_t_depth,
                                     hidden_state=self.num_hidden, init_strategy_map=self.init_strategy,
                                     activation=self.activation, zero_state=self.zero_state, name='revised_rnn',
                                     input_x=self.input_data_x, input_t=self.input_data_t)
        intensity_component = attention_mechanism.Intensity(time_stamp=self.time_stamps, batch_size=self.batch_size,
                                                            x_depth=self.input_x_depth, t_depth=self.input_t_depth,
                                                            mutual_intensity_path=self.mutual_intensity_path,
                                                            base_intensity_path=self.base_intensity_path,
                                                            name='intensity', placeholder_x=self.input_data_x,
                                                            placeholder_t=self.input_data_t)
        attention_component = attention_mechanism.AttentionMechanism(revised_rnn, intensity_component)
        self.c_weight, self.c_bias, self.r_weight, self.r_bias = self.__output_parameter()

        hidden_states_list = revised_rnn.states_tensor

        output_mix_hidden_state = []
        with tf.name_scope('attention'):
            for time_stamps in range(1, self.time_stamps + 1):
                weight = attention_component(time_stamp=time_stamps)
                hidden_states = tf.unstack(hidden_states_list)[0: time_stamps]
                state_list = []
                for i in range(0, time_stamps):
                    state_list.append(weight[i] * hidden_states[i])
                state_list = tf.convert_to_tensor(state_list, tf.float64)
                mix_state = tf.reduce_sum(state_list, axis=0)
                output_mix_hidden_state.append(mix_state)

        with tf.name_scope('prediction'):
            c_pred_list = []
            r_pred_list = []
            for state in output_mix_hidden_state:
                c_pred = tf.sigmoid(tf.matmul(state, self.c_weight) + self.c_bias)
                r_pred = tf.matmul(state, self.r_weight) + self.r_bias
                c_pred_list.append(c_pred)
                r_pred_list.append(r_pred)

        with tf.name_scope('loss'):
            pass

        with tf.name_scope('optimization'):
            pass

    def __output_parameter(self):
        # TODO 修改初始化策略
        with tf.variable_scope('pred_para', reuse=tf.AUTO_REUSE):
            c_weight = tf.get_variable(name='classification_weight', shape=[self.num_hidden, self.input_x_depth],
                                       initializer=tf.random_normal_initializer(), dtype=tf.float64)
            c_bias = tf.get_variable(name='classification_bias', shape=[self.input_x_depth],
                                     initializer=tf.zeros_initializer(), dtype=tf.float64)
            r_weight = tf.get_variable(name='regression_weight', shape=[self.num_hidden, self.input_t_depth],
                                       initializer=tf.random_normal_initializer(), dtype=tf.float64)
            r_bias = tf.get_variable(name='regression_bias', shape=[1, ],
                                     initializer=tf.zeros_initializer(), dtype=tf.float64)
        return c_weight, c_bias, r_weight, r_bias


def main():
    save_path = "D:\\PythonProject\\DiseaseProgression\\src\\model\\train"
    activation = tf.tanh
    init_map = dict()
    init_map['gate_weight'] = tf.random_normal_initializer(0, 1)
    init_map['gate_bias'] = tf.random_normal_initializer(0, 1)
    init_map['candidate_weight'] = tf.random_normal_initializer(0, 1)
    init_map['candidate_bias'] = tf.random_normal_initializer(0, 1)
    mi_path = ""
    bi_path = ""
    model_config = ModelConfiguration(learning_rate=0.001, max_train_steps=4, batch_size=3, x_depth=5, t_depth=1,
                                      time_stamps=6, num_hidden=7, cell_type='revised_gru', summary_save_path=save_path,
                                      c_r_ratio=1, activation=activation, init_strategy=init_map,
                                      mutual_intensity_path=mi_path, base_intensity_path=bi_path)
    AttentionBasedModel(model_config)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(save_path, sess.graph)


if __name__ == "__main__":
    main()
