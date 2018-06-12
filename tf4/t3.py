import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.constant([[1, 2], [1, 2]])
y = tf.constant([[1, 1], [1, 2]])

z = tf.add_n(x, y)
print(z)
