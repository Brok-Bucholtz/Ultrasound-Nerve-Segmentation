import tensorflow as tf


def _create_layer(layer_input, weight, bias):
    layer = tf.nn.conv2d(layer_input, weight, strides=[1, 1, 1, 1], padding='SAME')
    layer = tf.nn.bias_add(layer, bias)
    layer = tf.nn.relu(layer)
    return tf.nn.max_pool(
        layer,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')


def create_cnn(model_input, model_output, dropout, image_shape, n_classes):
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