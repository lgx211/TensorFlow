# coding:utf-8
import tensorflow as tf
import numpy as np

# 1,生成模拟数据集
BATCH_SIZE = 8
seed = 23455

# 基于seed种子生成随机数
rng = np.random.RandomState(seed)

# 随机数返回一个32行2列的矩阵
X = rng.rand(32, 2)

# 从矩阵X里面取一行，如果该行值的和小于1则赋值Y为1，否则为0
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]

y_ = tf.placeholder(tf.float32, shape=(None, 1))

# 2,前向传播
x = tf.placeholder(tf.float32, shape=(None, 2))

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 3,反向传播
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 4,会话训练
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    STEPS = 30
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        print("start", start)
        print("end", end)
        print("feed_dict", {x: X[start:end], y_: Y[start:end]})
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
