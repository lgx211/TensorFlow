import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tf4.mnist_forword as mnist_forword
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model"
MODEL_NAME = "mnist_model"


def backward(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_forword.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forword.OUTPUT_NODE])
    y = mnist_forword.forword(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    # tf.add_n：把一个列表的东西都依次加起来
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    trian_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([trian_step, ema_op]):
        trian_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([trian_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 5000 == 0:
                print("After %d steps , loss on trianing bacth is %g" % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


if __name__ == '__main__':
    mnist = input_data.read_data_sets("./date/", one_hot=True)
    backward(mnist)
