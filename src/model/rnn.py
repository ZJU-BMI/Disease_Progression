from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
# Test Data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


class ProposedModel(object):
    def __init__(self):
        # Training Parameters
        self.learning_rate = 0.001
        self.training_steps = 10000
        self.batch_size = 128
        self.display_step = 200

        # Network Parameters
        self.num_input = 28  # MNIST data input (img shape: 28*28)
        self.time_steps = 28  # timesteps
        self.num_hidden = 128  # hidden layer num of features
        self.num_classes = 10  # MNIST total classes (0-9 digits)

        self.direct_run_node = dict()
        self.x, self.y = self.data_define()

    def data_define(self):
        with tf.name_scope('data_attribute'):
            x = tf.placeholder("float", [None, self.time_steps, self.num_input])
        with tf.name_scope('data_label'):
            y = tf.placeholder("float", [None, self.num_classes])
        return x, y

    def __rnn_state(self, x):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        with tf.name_scope('rnn'):
            x = tf.unstack(x, self.time_steps, 1, name='x')

            # Define a lstm cell with tensorflow
            lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0, name='lstm_cell')

            # Get lstm cell output
            outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        return outputs

    def __rnn_output(self, output):
        with tf.variable_scope('rnn_output'):
            weights = tf.get_variable(name='weight', shape=[self.num_hidden, self.num_classes],
                                      initializer=tf.random_normal_initializer())
            biases = tf.get_variable(name='bias', shape=[self.num_classes],
                                     initializer=tf.random_normal_initializer())
        return tf.matmul(output[-1], weights) + biases

    def model_construct(self):
        logits = self.__rnn_output(self.__rnn_state(self.x))
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        with tf.name_scope('train'):
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y), name='loss')
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, name='optimizer')
            train_op = optimizer.minimize(loss_op, name='train')

        with tf.name_scope('prediction'):
            # Evaluate model (with test logits, for dropout to be disabled)
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('summary'):
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('loss_op', loss_op)
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
            train_writer = tf.summary.FileWriter('D:\\PythonProject\\DiseaseProgression\\src\\model\\train', sess.graph)
            sess.run(init)

            for step in range(1, self.training_steps + 1):
                batch_x, batch_y = mnist.train.next_batch(self.batch_size)
                # Reshape data to get 28 seq of 28 elements
                batch_x = batch_x.reshape((self.batch_size, self.time_steps, self.num_input))
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={self.x: batch_x, self.y: batch_y})
                if step % self.display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    merge, loss, acc = sess.run([summary, loss_op, accuracy], feed_dict={self.x: batch_x,
                                                                                         self.y: batch_y})
                    train_writer.add_summary(merge, step)

                    print("Step " + str(step) + ", Minibatch Loss= " +
                          "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

            print("Optimization Finished!")

            # Calculate accuracy for 128 mnist test images
            test_len = 128
            test_data = mnist.test.images[:test_len].reshape((-1, self.time_steps, self.num_input))
            test_label = mnist.test.labels[:test_len]
            print("Testing Accuracy:", sess.run(accuracy, feed_dict={self.x: test_data, self.y: test_label}))


def main():
    model = ProposedModel()
    model.model_construct()
    model.train()


if __name__ == "__main__":
    main()
