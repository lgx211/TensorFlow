import tensorflow as tf;
import numpy as np;

input1 = tf.constant([[1.0, 2.0, 3.0]])
input2 = tf.Variable([[1.0, 2.0, 3.0]])
a = tf.add_n([input1, input2])
b = tf.add_n(input1, input2)
c = tf.add_n(input1)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(a))
    print(sess.run(b))
