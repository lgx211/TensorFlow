import tensorflow as tf;
import numpy as np;

input1 = tf.constant([[1, 2], [1, 2]])
input2 = tf.constant([[1, 1], [1, 2]])

z = tf.add_n([input1, input2])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(z))
