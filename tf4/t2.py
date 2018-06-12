# encoding = utf-8
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 读取数据
mnist = input_data.read_data_sets('./date/', one_hot=True)

print("训练集个数：", mnist.train.num_examples)
print("验证集个数：", mnist.validation.num_examples)
print("测试集个数：", mnist.test.num_examples)

# 每个标签是一个一维数组，如下标为7的值为1，即该训练值有100%的概率为7
print("mnist训练集的第一个标签", mnist.train.labels[0])

# 图片信息。28*28的一维数组。
print("mnist训练集的第一个数据", mnist.train.images[0])
