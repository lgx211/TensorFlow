import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import time
import tf6.mnist_backword as mnist_backword
import tf6.mnist_forword as mnist_forword
import numpy as np

TEST_INTERVAL_SECS = 5


def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            mnist.test.num_examples,
            mnist_forword.IMAGE_SIZE,
            mnist_forword.IMAGE_SIZE,
            mnist_forword.NUM_CHANNELS
        ])
        y_ = tf.placeholder(tf.float32, [None, mnist_forword.OUTPUT_NODE])
        # 在测试程序中使用的是训练好的网络，故不使用 dropout，而是让所有神经元都参与运算，从而输出识别准确率。
        y = mnist_forword.forword(x, False, None)

        ema = tf.train.ExponentialMovingAverage(mnist_backword.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:

                # 断点续训
                ckpt = tf.train.get_checkpoint_state(mnist_backword.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s steps ,accuracy is %g " % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets('./date/', one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()
