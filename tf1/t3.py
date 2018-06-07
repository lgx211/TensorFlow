import tensorflow as tf

y = tf.random_normal([2, 3])

print(y)

with tf.Session() as sess:
    print(sess.run(y))
