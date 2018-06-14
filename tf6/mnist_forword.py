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


# x是输入描述。w是卷积层描述
# strides表示卷积核在不同维度上的移动步长为1，第一维和第四维一定是1，这是因为卷积层的步长只对矩阵的长和宽有效；
# padding='SAME'表示使用全0填充，而'VALID'表示不填充
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


# x是输入描述。ksize是池化层描述
# strides表示过滤器移动步长是2，'SAME'提供使用全0填充
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def forword(x, train, regularizer):
    # 1，实现第一层卷积
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)

    # 2，实现第二层卷积
    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    # 3，将第二层池化层的输出pool2矩阵转化为全连接层的输入格式即向量形式
    # 根据.get_shape()函数得到 pool2 输出矩阵的维度，并存入 list 中。
    pool_shape = pool2.get_shape().as_list()
    # 从 list 中依次取出矩阵的长宽及深度，并求三者的乘积，得到矩阵被拉长后的长度。
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 将pool2转换为一个batch的向量再传入后续的全连接。其中， pool_shape[0]为一个 batch 值。
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 4，实现第三层全连接层
    # 初始化全连接层的权重，并加入正则化。
    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    # 初始化全连接层的偏置项。
    fc1_b = get_bias([FC_SIZE])
    # 将转换后的reshaped向量与权重fc1_w做矩阵乘法运算，然后再加上偏置，最后再使用relu进行激活。
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # 如果是训练阶段，则对该层输出使用dropout，也就是随机的将该层输出中的一半神经元置为无效，是为了避免过拟合而设置的，一般只在全连接层中使用。
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5)

    # 5，实现第四层全连接层的前向传播过程
    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y
