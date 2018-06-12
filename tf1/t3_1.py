import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=2, mean=0, seed=1))
w2 = tf.Variable(tf.truncated_normal([2, 3], stddev=2, mean=0, seed=1))
w3 = tf.Variable(tf.random_uniform(shape=[2,3], minval=0, maxval=1, dtype=tf.float32, seed=1))

print("w1:", w1)
print("w2:", w2)
print("w3:", w3)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print("w1:", w1)
    print("w2:", w2)
    print("w3:", w3)

    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))
    print("w3:", sess.run(w3))
