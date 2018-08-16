# coding=utf-8
import numpy as np
import tensorflow as tf

data = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])
tensor = tf.placeholder('int32', [2, None, 4])
tensor_2 = tf.placeholder('int32', [2, None])
_, top_k = tf.nn.top_k(tensor, 2)

top_k_unstack = tf.unstack(top_k, axis=2)
for item in top_k_unstack:
    a = tensor_2[item]

with tf.Session() as sess:
    _, indices = sess.run(top_k, feed_dict={tensor: data})

print(indices)
