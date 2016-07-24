import tensorflow as tf
import cv2
import math
from sklearn import cross_validation
from sklearn.metrics import f1_score

from feature_extraction import get_detection_data


def _create_layer(layer_input, weight, bias):
    layer = tf.nn.conv2d(layer_input, weight, strides=[1, 1, 1, 1], padding='SAME')
    layer = tf.nn.bias_add(layer, bias)
    layer = tf.nn.relu(layer)
    return tf.nn.max_pool(
        layer,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')


def _create_cnn(model_input, model_output, dropout, image_shape, n_classes):
    learning_rate = 0.001

    model_input = tf.reshape(model_input, shape=[-1, image_shape[0], image_shape[1], 1])

    conv1 = _create_layer(
        model_input,
        tf.Variable(tf.random_normal([3, 3, 1, 16])),
        tf.Variable(tf.random_normal([16])))
    conv2 = _create_layer(
        conv1,
        tf.Variable(tf.random_normal([3, 3, 16, 32])),
        tf.Variable(tf.random_normal([32])))
    conv3 = _create_layer(
        conv2,
        tf.Variable(tf.random_normal([3, 3, 32, 64])),
        tf.Variable(tf.random_normal([64])))
    conv4 = _create_layer(
        conv3,
        tf.Variable(tf.random_normal([3, 3, 64, 128])),
        tf.Variable(tf.random_normal([128])))
    conv5 = _create_layer(
        conv4,
        tf.Variable(tf.random_normal([3, 3, 128, 256])),
        tf.Variable(tf.random_normal([256])))

    fc1 = tf.reshape(conv5, [-1, 1024])
    fc1 = tf.matmul(fc1, tf.Variable(tf.random_normal([1024, 256 * 16])))
    fc1 = tf.add(fc1, tf.Variable(tf.random_normal([256 * 16])))
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    pred = tf.matmul(fc1, tf.random_normal([256 * 16, n_classes]))
    pred = tf.add(pred, tf.random_normal([n_classes]))

    prediction = tf.argmax(pred, 1)

    cost = tf.nn.softmax_cross_entropy_with_logits(pred, model_output)
    cost = tf.reduce_mean(cost)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    return prediction, optimizer


def get_predictions(session, model, model_input, inputs):
    outputs = []
    for input in inputs:
        outputs.append(session.run(model, feed_dict={model_input: [input]})[0])
    return outputs


def run_cnn_detection():
    batch_size = 16
    image_width = 58  # 580
    image_height = 42  # 420
    n_classes = 2
    dropout = 0.75

    model_input = tf.placeholder(tf.float32, [None, image_width, image_height])
    model_output = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)

    x_all, y_all = get_detection_data()
    x_all = [cv2.resize(image, (image_height, image_width), interpolation=cv2.INTER_CUBIC) for image in x_all]
    y_all = [[float(y_element), float(not y_element)] for y_element in y_all]

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_all, y_all, test_size=0.25)

    prediction, optimizer = _create_cnn(model_input, model_output, dropout, (image_width, image_height), n_classes)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        print "Training CNN..."
        for batch_i in xrange(int(math.ceil(len(x_train)/float(batch_size)))):
            front_batch = batch_i * batch_size
            batch_x = x_train[front_batch: front_batch + batch_size]
            batch_y = y_train[front_batch: front_batch + batch_size]

            sess.run(optimizer, feed_dict={model_input: batch_x, model_output: batch_y, keep_prob: dropout})

        print "F1 score for train set: {}".format(f1_score(
            [y[1] for y in y_train],
            get_predictions(sess, prediction, model_input, x_train)))
        print "Predicting Test Set..."
        print "F1 score for test set: {}".format(f1_score(
            [y[1] for y in y_test],
            get_predictions(sess, prediction, model_input, x_test)))
