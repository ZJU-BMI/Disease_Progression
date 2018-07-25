# coding=utf-8


class ModelConfiguration(object):
    def __init__(self, x_depth, max_time_stamp, num_hidden, cell_type, init_map,
                 c_r_ratio, activation, init_strategy, mutual_intensity_path, base_intensity_path, zero_state,
                 file_encoding, time_decay_function, t_depth, threshold):
        """
        :param x_depth: defines the dimension of the input_x in a specific time stamp, it also indicates the number
        of type of event
        :param t_depth: defines the time of a specific time stamp, raise error if it is not 1
        :param max_time_stamp: should be a scalar, the length of RNN
        :param num_hidden: should be a scalar, the dimension of a hidden state
        :param cell_type: should be a string, 'revised_gru' or 'gru'
        :param c_r_ratio: should be a scalar, the coefficient to adjust the weight between classification task and
        regression task.
        :param zero_state: the zero state of rnn, np.ndarray with shape [num_hidden,]
        :param activation: should be a function object, activation function of RNN
        :param init_strategy: parameter initialize strategy for every parameter
        :param mutual_intensity_path: a file path, reading the information of mutual intensity
        :param base_intensity_path: a file path, reading the information of base intensity
        :param file_encoding: intensity file encoding
        :param time_decay_function: which is long (at least 10,000 elements) 1-d np.ndarray, each entry indicates the
        intensity of corresponding time stamps
        :param threshold: threshold for metrics
        """

        self.time_decay_function = time_decay_function

        # Model Parameters
        self.c_r_ratio = c_r_ratio
        self.input_x_depth = x_depth
        self.input_t_depth = t_depth
        self.max_time_stamp = max_time_stamp

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

        # parameter initializer
        self.init_map = init_map
        self.threshold = threshold


class TrainingConfiguration(object):
    def __init__(self, learning_rate, optimizer, weight_decay, train_save_path, test_save_path, batch_size, iteration):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.train_save_path = train_save_path
        self.test_save_path = test_save_path
        self.batch_size = batch_size
        self.iteration = iteration
