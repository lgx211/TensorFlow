import tensorflow as tf
import numpy as np
from PIL import Image
import tf5.mnist_backword as mnist_backword
import tf5.mnist_forword as mnist_forword


# 图片预处理
def pre_pic(picName):
    # 打开要识别的图片
    img = Image.open(picName)
    # 用消除锯齿的方法，把原图片转为28*28像素的图片
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    # 将图片转为灰度图。然后转为矩阵格式
    im_arr = np.array(reIm.convert('L'))
    threshold = 50
    # 白底黑字转为要求所需的黑底白字
    for i in range(28):
        for j in range(28):
            # 颜色反转
            im_arr[i][j] = 255 - im_arr[i][j]
            # 去掉噪点。低于阈值认为是纯黑色。高于阈值认为是纯白色
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    # 整理形状为1行784列的矩阵数据
    nm_arr = im_arr.reshape([1, 784])
    # 把数据转为浮点型
    nm_arr = nm_arr.astype(np.float32)
    # 把0-255之间的图片颜色数据转为0-1之间的浮点型数据
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)

    return img_ready


# 重现计算图
def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forword.INPUT_NODE])
        y = mnist_forword.forword(x, None)

        # 预测结果值
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_backword.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backword.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


def application():
    testNum = input("请输入本次要识别的图片张数：")
    for i in range(int(testNum)):
        testPic = input("图片路径：")
        # 把图片信息转为数组
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)

        print("该图片的值是：", preValue)

if __name__ == '__main__':
    application()