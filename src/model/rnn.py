from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data


class ProposedModel(object):
    def __init__(self, data, model_config):
        # dataset description
        self.training_data = data.training_data
        self.training_label = data.training_label
        self.test_data = data.test_data
        self.test_label = data.test_label
        self.training_time = data.training_time
        self.test_time = data.test_time

        self.save_path = model_config.save_path

        # Training Parameters
        self.learning_rate = model_config.learning_rate
        self.training_steps = model_config.max_training_steps
        self.batch_size = model_config.batch_size

        # Network Parameters
        self.num_input = model_config.rnn_num_input
        self.time_steps = model_config.rnn_time_steps
        self.num_hidden = model_config.rnn_num_hidden
        self.num_classes = model_config.rnn_num_classes
        self.cell_type = model_config.cell_type
        self.c_r_ratio = model_config.c_r_ratio

        self.direct_run_node = dict()
        self.x, self.y, self.t = self.data_define()

    def data_define(self):
        with tf.name_scope('data_attribute'):
            x = tf.placeholder("float", [None, self.time_steps, self.num_input])
        with tf.name_scope('data_label'):
            y = tf.placeholder("float", [None, self.time_steps, self.num_classes])
        with tf.name_scope('data_time'):
            t = tf.placeholder("float", [None, self.time_steps, 1])
        return x, y, t

    def __rnn_state(self, x, cell_type):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, time_steps, n_input)
        # Required shape: 'time_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'time_steps' tensors of shape (batch_size, n_input)
        with tf.name_scope('rnn'):
            x = tf.unstack(x, self.time_steps, 1, name='x')

            if cell_type == 'lstm' or cell_type == 'gru' or cell_type == 'basic':
                if cell_type == 'lstm':
                    cell = rnn.LSTMCell(self.num_hidden, forget_bias=1.0, name='lstm')
                elif cell_type == 'gru':
                    cell = rnn.GRUCell(self.num_hidden, name='gru')
                else:
                    cell = rnn.BasicRNNCell(self.num_hidden, name='basic')
            else:
                raise Exception('cell type need to be "lstm", "gru", "basic"')

            # Get lstm cell output
            states, final_state = rnn.static_rnn(cell, x, dtype=tf.float32)

            return states

    def __rnn_output(self, state):

        with tf.variable_scope('rnn_output_para') as scope:
            c_weight = tf.get_variable(name='classification_weight', shape=[self.num_hidden, self.num_classes],
                                       initializer=tf.random_normal_initializer())
            c_bias = tf.get_variable(name='classification_bias', shape=[self.num_classes],
                                     initializer=tf.zeros_initializer())
            r_weight = tf.get_variable(name='regression_weight', shape=[self.num_hidden, 1],
                                       initializer=tf.random_normal_initializer())
            r_bias = tf.get_variable(name='regression_bias', shape=[1, ],
                                     initializer=tf.zeros_initializer())
        with tf.name_scope('output_sequence'):
            state_list = tf.split(state, len(state), axis=0, name='rnn_output_split')
            classification_output_list = []
            regression_output_list = []

            scope.reuse_variables()
            for item in state_list:
                item = item[0]
                c_output = tf.matmul(item, c_weight) + c_bias
                r_output = tf.matmul(item, r_weight) + r_bias
                classification_output_list.append(c_output)
                regression_output_list.append(r_output)
            regression_output_list = tf.reshape(regression_output_list, [-1, self.time_steps, 1])
            classification_output_list = tf.reshape(classification_output_list, [-1, self.time_steps, self.num_classes])
        return classification_output_list, regression_output_list

    def model_construct(self):
        c_output_list, r_output_list = self.__rnn_output(self.__rnn_state(self.x, self.cell_type))

        with tf.name_scope('prediction'):
            # softmax will normalize the value of entries in same vector
            c_pred = tf.nn.softmax(c_output_list)
            r_pred = r_output_list
            correct_class_prediction = tf.equal(tf.argmax(c_pred, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_class_prediction, tf.float32))
            absolute_error = tf.reduce_mean(tf.abs(r_pred - self.t))

            # absolute_time_error = tf.

        # Define loss and optimizer
        with tf.name_scope('train'):
            # TODO
            # 此处对y的处理在之后要修改，在minist中，y,time是恒定的，但是在目标任务中，每个time step的输出都应当有差异
            with tf.name_scope('classification_loss'):
                loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=c_output_list, labels=self.y))
            with tf.name_scope('regression_loss'):
                loss_r = tf.losses.mean_squared_error(labels=self.t, predictions=r_pred)

            loss_op = loss_c + self.c_r_ratio * loss_r

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, name='optimizer')
            train_op = optimizer.minimize(loss_op, name='train')

        with tf.name_scope('summary'):
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('time_absolute_error', absolute_error)
            tf.summary.scalar('loss_all', loss_op)
            tf.summary.scalar('loss_classification', loss_c)
            tf.summary.scalar('loss_regression', loss_r)
            tf.summary.scalar('loss_regression', loss_r)
            merged = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        self.direct_run_node['initialize'] = init
        self.direct_run_node['train'] = train_op
        self.direct_run_node['accuracy'] = accuracy
        self.direct_run_node['summary'] = merged
        self.direct_run_node['loss'] = loss_op

    def train(self):
        init = self.direct_run_node['initialize']
        train_op = self.direct_run_node['train']
        accuracy = self.direct_run_node['accuracy']
        summary = self.direct_run_node['summary']
        loss_op = self.direct_run_node['loss']

        # Start training
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(self.save_path, sess.graph)

            sess.run(init)
            for step in range(1, self.training_steps + 1):
                batch_x = self.training_data[step % len(self.training_data)]
                batch_y = self.training_label[step % len(self.training_label)]
                batch_t = self.training_time[step % len(self.training_time)][0]

                # Reshape data
                batch_x = batch_x.reshape((self.batch_size, self.time_steps, self.num_input))
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={self.x: batch_x, self.y: batch_y, self.t: batch_t})
                if step % 200 == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    merge, loss, acc = sess.run([summary, loss_op, accuracy], feed_dict={self.x: batch_x,
                                                                                         self.y: batch_y,
                                                                                         self.t: batch_t})
                    train_writer.add_summary(merge, step)

                    print("Step " + str(step) + ", Minibatch Loss= " +
                          "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

            print("Optimization Finished!")
            # print("Testing Accuracy:", sess.run(accuracy, feed_dict={self.x: self.test_data,
            # self.y: self.test_label}))


class ModelConfig(object):
    def __init__(self, learning_rate, max_train_steps, batch_size, rnn_num_input, rnn_time_steps, rnn_num_hidden,
                 rnn_num_classes, cell_type, save_path, c_r_ratio):
        self.save_path = save_path
        self.c_r_ratio = c_r_ratio

        # Training Parameters
        self.learning_rate = learning_rate
        self.max_training_steps = max_train_steps
        self.batch_size = batch_size

        # Network Parameters
        self.rnn_num_input = rnn_num_input
        self.rnn_time_steps = rnn_time_steps
        self.rnn_num_hidden = rnn_num_hidden
        self.rnn_num_classes = rnn_num_classes
        self.cell_type = cell_type


class TestData(object):
    def __init__(self, data_path, time_steps, num_input, test_len, batch_size):
        data = input_data.read_data_sets(data_path, one_hot=True)
        test_data = data.test.images[:test_len].reshape((-1, time_steps, num_input))
        test_label = data.test.labels[:test_len]
        test_time = np.ones([len(test_label)]).tolist()
        training_data = []
        training_label = []
        training_time = []

        for i in range(0, 400):
            training_data_minibatch, training_label_minibatch = data.train.next_batch(batch_size)
            training_data.append(training_data_minibatch)

            training_label_minibatch = np.expand_dims(training_label_minibatch, axis=2)
            training_label_minibatch = np.repeat(training_label_minibatch, [time_steps], axis=2)
            training_label_minibatch = np.transpose(training_label_minibatch, [0, 2, 1])
            training_label.append(training_label_minibatch)

            training_time.append(
                [np.full([training_label_minibatch.shape[0], training_label_minibatch.shape[1], 1], 1.3)])

        self.training_data = training_data
        self.training_label = training_label
        self.training_time = training_time
        self.test_data = test_data
        self.test_label = test_label
        self.test_time = test_time


def main():
    save_path = "D:\\PythonProject\\DiseaseProgression\\src\\model\\train"
    data_path = "/tmp/data/"
    time_steps = 28
    num_input = 28
    c_r_ratio = 1

    data = TestData(data_path=data_path, time_steps=time_steps, num_input=num_input, test_len=128, batch_size=128)
    model_config = ModelConfig(learning_rate=0.001, max_train_steps=20, batch_size=128, rnn_num_input=num_input,
                               rnn_time_steps=28, rnn_num_hidden=time_steps, rnn_num_classes=10, cell_type='gru',
                               save_path=save_path, c_r_ratio=c_r_ratio)

    model = ProposedModel(data=data, model_config=model_config)
    model.model_construct()
    model.train()


if __name__ == "__main__":
    main()
