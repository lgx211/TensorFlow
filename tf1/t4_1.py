import tensorflow as tf

# 占一个一行两列的二维数组
x = tf.placeholder(tf.float32, shape=[1, 2])

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    print(sess.run(init_op))
    print(sess.run(y, feed_dict={x: [[0.7, 0.5]]}))
