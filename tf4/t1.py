import tensorflow as tf

x = [[1, 2], [2, 2]]

with tf.Session() as sess:
    print(sess.run(tf.reduce_mean(x, 1)))
