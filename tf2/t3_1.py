# coding:utf-8
import tensorflow as tf
import numpy as np

w = tf.Variable(tf.constant(5, dtype=tf.float32))

loss = tf.square(w + 1)

# 学习率过大，结果不收敛
train_step = tf.train.GradientDescentOptimizer(1).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)

        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After %s steps : w is %f ,loss is %f" % (i, w_val, loss_val))
