import tensorflow as tf

# 1,生成随机数
# 表示生成正态分布随机数，形状两行三列，标准差是 2，均值是 0，随机种子是 1。
w1 = tf.random_normal([2, 3], stddev=2, mean=0, seed=1)

# 表示去掉偏离过大的正态分布，也就是如果随机出来的数据偏离平均值超过两个标准差，这个数据将重新生成。
w2 = tf.truncated_normal([2, 3], stddev=2, mean=0, seed=1)

# 表示从一个均匀分布[minval maxval)中随机采样，注意定义域是左闭右开，即包含minval，不包含 maxval。
w3 = tf.random_uniform(shape=[2, 3], minval=0, maxval=1, dtype=tf.float32, seed=1)

print("w1:", w1)
print("w2:", w2)
print("w3:", w3)

with tf.Session() as sess:
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))
    print("w3:", sess.run(w3))
