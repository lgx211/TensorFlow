import tensorflow as tf

# 加法 计算图
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a + b
d = tf.add(a, b)
print(a)
print(b)
print(c)
print(d)

# 乘法 计算图
x = tf.constant([[1, 3]])
w = tf.constant([[1], [4]])
y = tf.matmul(x, w)
print(y)

# 计算 会话
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(y))
