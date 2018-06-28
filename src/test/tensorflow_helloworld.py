import tensorflow as tf

hello = tf.constant('hello deep learning')
sess = tf.Session()
print(sess.run(hello))

