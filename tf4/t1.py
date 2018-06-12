import tensorflow as tf

x = [[1., 2.], [10., 10.]]
y = [[[1., 2.], [10., 10.]], [[1., 2.], [10., 10.]]]

with tf.Session() as sess:
    # 张量x所有元素的平均值
    print("张量x所有元素的平均值", sess.run(tf.reduce_mean(x)))
    # 张量x每行的平均值
    print("张量x每行的平均值", sess.run(tf.reduce_mean(x, 1)))
    # 张量x每列的平均值
    print("张量x每列的平均值", sess.run(tf.reduce_mean(x, 0)))

    print("张量y所有元素的平均值",sess.run(tf.reduce_mean(y)))
    print("张量y二维的平均值",sess.run(tf.reduce_mean(y, 1)))
    print("张量y一维的平均值",sess.run(tf.reduce_mean(y, 0)))

