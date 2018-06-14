import tensorflow as tf

# 定义神经网络可以接收的图片的尺寸和通道数
IMAGE_SIZE = 28
NUM_CHANNELS = 1

# 定义第一层卷积核的大小和个数
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32

# 定义第二层卷积核的大小和个数
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64

# 定义第三层全连接层的神经元个数
FC_SIZE = 512
# 定义第四层全连接层的神经元个数
OUTPUT_NODE = 10


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


# x，batch数。  w，卷积层的权重
# strides表示卷积核在不同维度上的移动步长为1，第一维和第四维一定是1，这是因为卷积层的步长只对矩阵的长和宽有效；
# padding='SAME'表示使用全0 填充，而'VALID'表示不填充
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
