# coding=utf-8
import auc as auc_revise
import tensorflow as tf

x_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.8, 0.9, 1]
y_1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
x_2 = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
y_2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

x_placeholder = tf.placeholder(tf.float64, [10])
y_placeholder = tf.placeholder(tf.bool, [10])
auc = tf.metrics.auc(labels=y_placeholder, predictions=x_placeholder)
initializer = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(initializer)
    for i in range(3):
        auc_value, acc_op = sess.run(auc, feed_dict={x_placeholder: x_1, y_placeholder: y_1})
        print('auc_2: ' + str(auc_value) + ", update_op: " + str(acc_op))
        auc_value, acc_op = sess.run(auc, feed_dict={x_placeholder: x_2, y_placeholder: y_2})
        print('auc_1: ' + str(auc_value) + ", update_op: " + str(acc_op))

auc_revise = auc_revise.auc(labels=y_placeholder, predictions=x_placeholder)
initializer = tf.group(tf.global_variables_initializer())
with tf.Session() as sess:
    sess.run(initializer)
    for i in range(3):
        auc_value = sess.run(auc_revise, feed_dict={x_placeholder: x_2, y_placeholder: y_2})
        print('auc_1: ' + str(auc_value))
        auc_value = sess.run(auc_revise, feed_dict={x_placeholder: x_1, y_placeholder: y_1})
        print('auc_2: ' + str(auc_value))
