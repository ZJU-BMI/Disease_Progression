import numpy as np
import tensorflow as tf

import attention_mechanism
import rnn


class ModelConfiguration(object):
    def __init__(self, learning_rate, batch_size, x_depth, time_stamps, num_hidden, cell_type, summary_save_path,
                 c_r_ratio, activation, init_strategy, mutual_intensity_path, base_intensity_path, zero_state,
                 file_encoding, t_depth=1):
        """
        :param learning_rate: should be a scalar
        :param batch_size: the size of minibatch, a scalar
        :param x_depth: defines the dimension of the input_x in a specific time stamp, it also indicates the number
        of type of event
        :param t_depth: defines the time of a specific time stamp, raise error if it is not 1
        :param time_stamps: should be a scalar, the length of RNN
        :param num_hidden: should be a scalar, the dimension of a hidden state
        :param cell_type: should be a string, 'revised_gru' or 'gru'
        :param summary_save_path: a folder path, used to store information of tensorboard
        :param c_r_ratio: should be a scalar, the coefficient to adjust the weight between classification task and
        regression task.
        :param zero_state: the zero state of rnn, np.ndarray with shape [num_hidden,]
        :param activation: should be a function object, activation function of RNN
        :param init_strategy: parameter initialize strategy for every parameter
        :param mutual_intensity_path: a file path, reading the information of mutual intensity
        :param base_intensity_path: a file path, reading the information of base intensity
        :param file_encoding: intensity file encoding
        """
        # Tensorboard Data And Output Save Path
        self.model_summary_save_path = summary_save_path

        # Training Parameters
        self.learning_rate = learning_rate
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
        self.zero_state = zero_state
        self.init_strategy = init_strategy

        # Attention Parameters
        self.mutual_intensity_path = mutual_intensity_path
        self.base_intensity_path = base_intensity_path
        self.file_encoding = file_encoding


class AttentionBasedModel(object):
    def __init__(self, model_config):
        # Tensorboard Data And Output Save Path
        self.__model_summary_save_path = model_config.model_summary_save_path

        # Training Parameters
        self.__learning_rate = model_config.learning_rate
        self.__batch_size = model_config.batch_size

        # General Model Parameters
        self.__c_r_ratio = model_config.c_r_ratio
        self.__input_x_depth = model_config.input_x_depth
        self.__input_t_depth = model_config.input_t_depth
        self.__time_stamps = model_config.time_stamps

        # Network Parameters
        self.__num_hidden = model_config.num_hidden
        self.__cell_type = model_config.cell_type
        self.__activation = model_config.activation
        self.__zero_state = model_config.zero_state
        self.__init_strategy = model_config.init_strategy

        # Attention Parameters
        self.__mutual_intensity_path = model_config.mutual_intensity_path
        self.__base_intensity_path = model_config.base_intensity_path
        self.__file_encoding = model_config.file_encoding

        # Output Parameters
        self.__c_weight = None
        self.__c_bias = None
        self.__r_weight = None
        self.__r_bias = None

        # data feed node
        self.input_data_x = None
        self.input_data_t = None

        # expose_node
        self.merge_summary = None
        self.optimize = None

        self.build()

    def build(self):
        with tf.name_scope('input_data'):
            self.input_data_x = tf.placeholder('float64', shape=[self.__time_stamps, self.__batch_size,
                                                                 self.__input_x_depth], name='input_x')
            self.input_data_t = tf.placeholder('float64', shape=[self.__time_stamps, self.__batch_size,
                                                                 self.__input_t_depth], name='input_t')

        revised_rnn = rnn.RevisedRNN(time_stamp=self.__time_stamps, batch_size=self.__batch_size,
                                     x_depth=self.__input_x_depth, t_depth=self.__input_t_depth,
                                     hidden_state=self.__num_hidden, init_strategy_map=self.__init_strategy,
                                     activation=self.__activation, zero_state=self.__zero_state,
                                     input_x=self.input_data_x, input_t=self.input_data_t)
        intensity_component = attention_mechanism.Intensity(time_stamp=self.__time_stamps, batch_size=self.__batch_size,
                                                            x_depth=self.__input_x_depth, t_depth=self.__input_t_depth,
                                                            mutual_intensity_path=self.__mutual_intensity_path,
                                                            base_intensity_path=self.__base_intensity_path,
                                                            name='intensity', placeholder_x=self.input_data_x,
                                                            placeholder_t=self.input_data_t,
                                                            file_encoding=self.__file_encoding,
                                                            para_init_map=self.__init_strategy)

        attention_component = attention_mechanism.AttentionMechanism(revised_rnn, intensity_component)
        self.__c_weight, self.__c_bias, self.__r_weight, self.__r_bias = self.__output_parameter()

        hidden_states_list = revised_rnn.states_tensor

        output_mix_hidden_state = []
        with tf.name_scope('attention'):
            for time_stamps in range(1, self.__time_stamps + 1):
                with tf.name_scope('mix_state'):
                    with tf.name_scope('weight'):
                        weight = attention_component(time_stamp=time_stamps)
                    with tf.name_scope('states'):
                        hidden_states = tf.unstack(hidden_states_list)[0: time_stamps]
                    state_list = []
                    with tf.name_scope('state_mix'):
                        for i in range(0, time_stamps):
                            state_list.append(weight[i] * hidden_states[i])
                        state_list = tf.convert_to_tensor(state_list, tf.float64)
                        mix_state = tf.reduce_sum(state_list, axis=0)
                        output_mix_hidden_state.append(mix_state)
            output_mix_hidden_state = tf.convert_to_tensor(output_mix_hidden_state, dtype=tf.float64,
                                                           name='attention_states')

        with tf.name_scope('output'):
            output_mix_hidden_state = tf.unstack(output_mix_hidden_state, axis=0)
            c_pred_list = []
            r_pred_list = []
            with tf.name_scope('c_output'):
                for state in output_mix_hidden_state:
                    c_pred = tf.matmul(state, self.__c_weight) + self.__c_bias
                    c_pred_list.append(c_pred)
                c_pred_list = tf.convert_to_tensor(c_pred_list)
            with tf.name_scope('r_output'):
                for state in output_mix_hidden_state:
                    r_pred = tf.matmul(state, self.__r_weight) + self.__r_bias
                    r_pred_list.append(r_pred)
                r_pred_list = tf.convert_to_tensor(r_pred_list)

        with tf.name_scope('normal_pred'):
            self.__performance_measure(c_pred_list, r_pred_list)

        with tf.name_scope('loss'):
            with tf.name_scope('c_loss'):
                c_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_data_x,
                                                                               logits=c_pred_list))
            with tf.name_scope('r_loss'):
                r_loss = tf.reduce_sum(tf.cast(tf.losses.mean_squared_error(labels=self.input_data_t,
                                                                            predictions=r_pred_list),
                                               dtype=tf.float64))
            with tf.name_scope('loss_sum'):
                loss_sum = c_loss + self.__c_r_ratio * r_loss

            with tf.name_scope('summary'):
                tf.summary.scalar('loss_c', c_loss)
                tf.summary.scalar('loss_r', r_loss)
                tf.summary.scalar('loss_sum', loss_sum)

        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer()
            self.optimize = optimizer.minimize(loss_sum)

        self.merge_summary = tf.summary.merge_all()

    def __output_parameter(self):
        with tf.variable_scope('pred_para', reuse=tf.AUTO_REUSE):
            c_weight = tf.get_variable(name='classification_weight', shape=[self.__num_hidden, self.__input_x_depth],
                                       initializer=self.__init_strategy["classification_weight"], dtype=tf.float64)
            c_bias = tf.get_variable(name='classification_bias', shape=[self.__input_x_depth],
                                     initializer=self.__init_strategy["classification_bias"], dtype=tf.float64)
            r_weight = tf.get_variable(name='regression_weight', shape=[self.__num_hidden, self.__input_t_depth],
                                       initializer=self.__init_strategy["regression_weight"], dtype=tf.float64)
            r_bias = tf.get_variable(name='regression_bias', shape=[1, ],
                                     initializer=self.__init_strategy["regression_bias"], dtype=tf.float64)
        return c_weight, c_bias, r_weight, r_bias

    def __performance_measure(self, c_pred_list, r_pred_list):
        # TODO, 加一个分门别类的测试精度，需要包括Recall, Specific, AUC, PR等
        pass


def main():
    save_path = "D:\\PythonProject\\DiseaseProgression\\src\\model\\train"
    activation = tf.tanh
    init_map = dict()
    init_map['gate_weight'] = tf.random_normal_initializer(0, 1)
    init_map['gate_bias'] = tf.random_normal_initializer(0, 1)
    init_map['candidate_weight'] = tf.random_normal_initializer(0, 1)
    init_map['candidate_bias'] = tf.random_normal_initializer(0, 1)
    init_map['classification_weight'] = tf.random_normal_initializer(0, 1)
    init_map['classification_bias'] = tf.random_normal_initializer(0, 1)
    init_map['regression_weight'] = tf.random_normal_initializer(0, 1)
    init_map['regression_bias'] = tf.random_normal_initializer(0, 1)
    init_map['mutual_intensity'] = tf.random_normal_initializer(0, 1)
    init_map['base_intensity'] = tf.random_normal_initializer(0, 1)

    num_hidden = 3
    batch_size = 8
    x_depth = 6
    t_depth = 1
    time_stamps = 4
    batch_count = 5
    cell_type = 'revised_gru'
    zero_state = np.random.normal(0, 1, [num_hidden, ])
    mi_path = "D:\\PythonProject\\DiseaseProgression\\resource\\mutual_intensity_sample.csv"
    bi_path = "D:\\PythonProject\\DiseaseProgression\\resource\\base_intensity_sample.csv"
    file_encoding = 'utf-8-sig'
    c_r_ratio = 1
    learning_rate = 0.001

    model_config = ModelConfiguration(learning_rate=learning_rate, batch_size=batch_size, x_depth=x_depth,
                                      t_depth=t_depth, time_stamps=time_stamps, num_hidden=num_hidden,
                                      cell_type=cell_type, summary_save_path=save_path, c_r_ratio=c_r_ratio,
                                      activation=activation, init_strategy=init_map, zero_state=zero_state,
                                      mutual_intensity_path=mi_path, base_intensity_path=bi_path,
                                      file_encoding=file_encoding)

    attention_model = AttentionBasedModel(model_config)
    init = tf.global_variables_initializer()

    x = np.random.normal(0, 1, [batch_count, time_stamps, batch_size, x_depth])
    t = np.random.normal(0, 1, [batch_count, time_stamps, batch_size, t_depth])

    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(save_path, sess.graph)
        for i in range(0, batch_count):
            input_x = x[i]
            input_t = t[i]
            merge, _ = sess.run([attention_model.merge_summary, attention_model.optimize],
                                feed_dict={attention_model.input_data_x: input_x,
                                           attention_model.input_data_t: input_t})
            train_writer.add_summary(merge, i)


if __name__ == "__main__":
    main()
