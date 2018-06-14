import tensorflow as tf
import numpy as np
from PIL import Image
import os

image_train_path = './mnist_data_jpg/mnist_train_jpg_60000/'
label_train_path = './mnist_data_jpg/mnist_train_jpg_60000.txt'
tfRecord_train = './data/mnist_train.tfrecords'

image_test_path = './mnist_data_jpg/mnist_test_jpg_10000/'
label_test_path = './mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_test = './data/mnist_test.tfrecords'

data_path = './data'
resize_height = 28
resize_weight = 28


# 生成tfRecord文件
def write_tfRecord(tfRecordName, image_path, label_path):
    # 新建一个 writer
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    # 计数器，显示进度
    num_pic = 0
    # 以读的形式打开标签文件，此处我的标签文件是一个TXT文件
    f = open(label_path, 'r')
    # 读取文件的内容，并赋值
    contents = f.readlines()
    # 关闭标签文件
    f.close()
    # 循环遍历txt文件中每行数据
    for content in contents:
        # 因为每行中每列的数据是以空格间隔，所以以空格拆分每列的数据
        value = content.split()
        # 图片路径 + 图片名字
        img_path = image_train_path + value[0]
        # 打开该图片
        img = Image.open(img_path)
        # 把图片转为二进制格式
        img_raw = img.tobytes()

        lables = [0] * 10
        lables[int(value[1])] = 1

        example = tf.train.Example(features=tf.train.Feature(features={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=lables))
        }))
        # 把信息序列化
        writer.write(example.SerializeToString())
        # 每处理完一次，计数器加1
        num_pic += 1
        print("the number of picture is :", num_pic)
    writer.close()
    print("write tfRecord successful")


# 读取tfRecord文件
def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path])
    reader = tf.TFRecordReader()
    # 解序列化
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([10], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       }
                                       )
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # 1行784列
    img.set_shape([784])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label


# 批量获取trRecord文件
def get_tfRecord(num, isTrain=True):
    # 如果读取训练集
    if isTrain:
        tfRecord_path = tfRecord_train
    # 如果读取训练集
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=num,
                                                    num_threads=2,
                                                    capacity=1000,
                                                    min_after_dequeue=700
                                                    )
    return img_batch, label_batch


def generate_tfRecord():
    # 判断路径是否存在
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print("the directory already create success")
    else:
        print("the directory is already exists")
    # 把训练集中的图片与标签放入名为“tfRecord_train”中tfRecord文件
    write_tfRecord(tfRecord_train, image_train_path, label_train_path)
    # 把测试集中的图片与标签放入名为“tfRecord_test”中tfRecord文件
    write_tfRecord(tfRecord_test, image_test_path, label_test_path)


if __name__ == '__main__':
    generate_tfRecord()
