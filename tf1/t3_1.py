import tensorflow as tf

w = tf.Variable(tf.random_normal([2, 3]))

print(w)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w))
