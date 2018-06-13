import tensorflow as tf

# 代码就是不断更新w的值，优化w的值。
w1 = tf.Variable(0, dtype=tf.float32)

# 计数器，轮数，不计入
global_step = tf.Variable(0, trainable=False)
# 平均移动衰减率
MOVING_AVERAGE_DECAY = 0.99
# 实例化滑动平均类
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

# 自动将所有待训练的参数汇总为列表
ema_op = ema.apply(tf.trainable_variables())

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 更新参数w1的值为1
    sess.run(tf.assign(w1, 1))
    # 更新参数后，运算
    sess.run(ema_op)
    # 打印运算过后的w1和w1滑动平均值的初始值
    print("第二次", sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
