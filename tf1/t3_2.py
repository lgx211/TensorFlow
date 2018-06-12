import tensorflow as tf

# 2,生成常量
# 表示生成[[0,0],[0,0],[0,0]]
x1 = tf.zeros([3, 2], tf.int32)

# 表示生成[[1,1],[1,1],[1,1]；
x2 = tf.ones([3, 2], tf.int32)

# 表示生成[[6,6],[6,6],[6,6]]；
x3 = tf.fill([3, 2], 6)

# 表示生成[3,2,1]
x4 = tf.constant([3, 2, 1])

print("x1:", x1)
print("x2:", x2)
print("x3:", x3)
print("x4:", x4)

with tf.Session() as sess:
    print("x1:", sess.run(x1))
    print("x2:", sess.run(x2))
    print("x3:", sess.run(x3))
    print("x4:", sess.run(x4))
