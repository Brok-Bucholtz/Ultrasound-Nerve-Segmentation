import tensorflow as tf


def _create_maxpool_layer(layer_input, weight, bias):
    layer = tf.nn.conv2d(layer_input, weight, strides=[1, 1, 1, 1], padding='SAME')
    layer = tf.nn.bias_add(layer, bias)
    layer = tf.nn.relu(layer)
    return tf.nn.max_pool(
        layer,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')


def _create_transpose_layer(layer_input, output_shape, transpose_weight, bias):
    batch_size = tf.shape(layer_input)[0]
    layer = tf.nn.conv2d_transpose(
        layer_input,
        transpose_weight,
        [batch_size, output_shape[0], output_shape[1], output_shape[2]],
        strides=[1, 2, 2, 1])
    layer = tf.nn.bias_add(layer, bias)
    return tf.nn.relu(layer)


def create_cnn(model_input, dropout, image_shape, resize_dividend, n_classes):
    model_input = tf.reshape(model_input, shape=[-1, image_shape[0], image_shape[1], 1])
    tf.image_summary('Input', model_input)
    model_input = tf.image.resize_images(
        model_input,
        image_shape[0]/resize_dividend,
        image_shape[1]/resize_dividend,
        tf.image.ResizeMethod.BICUBIC)
    tf.image_summary('Resize', model_input)

    conv1 = _create_maxpool_layer(
        model_input,
        tf.Variable(tf.random_normal([3, 3, 1, 16])),
        tf.Variable(tf.random_normal([16])))
    conv2 = _create_maxpool_layer(
        conv1,
        tf.Variable(tf.random_normal([3, 3, 16, 32])),
        tf.Variable(tf.random_normal([32])))
    conv3 = _create_maxpool_layer(
        conv2,
        tf.Variable(tf.random_normal([3, 3, 32, 64])),
        tf.Variable(tf.random_normal([64])))
    conv4 = _create_maxpool_layer(
        conv3,
        tf.Variable(tf.random_normal([3, 3, 64, 128])),
        tf.Variable(tf.random_normal([128])))
    conv5 = _create_maxpool_layer(
        conv4,
        tf.Variable(tf.random_normal([3, 3, 128, 256])),
        tf.Variable(tf.random_normal([256])))
    conv6 = _create_transpose_layer(
        conv5,
        [4, 4, 128],
        tf.Variable(tf.random_normal([2, 2, 128, 256])),
        tf.Variable(tf.random_normal([128])))
    conv7 = _create_transpose_layer(
        conv6,
        [8, 8, 64],
        tf.Variable(tf.random_normal([4, 4, 64, 128])),
        tf.Variable(tf.random_normal([64])))
    conv8 = _create_transpose_layer(
        conv7,
        [16, 16, 32],
        tf.Variable(tf.random_normal([8, 8, 32, 64])),
        tf.Variable(tf.random_normal([32])))
    conv9 = _create_transpose_layer(
        conv8,
        [32, 32, 16],
        tf.Variable(tf.random_normal([16, 16, 16, 32])),
        tf.Variable(tf.random_normal([16])))

    fc1 = tf.reshape(conv9, [-1, 32*32*16])
    fc1 = tf.matmul(fc1, tf.Variable(tf.random_normal([32*32*16, 16])))
    fc1 = tf.add(fc1, tf.Variable(tf.random_normal([16])))
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.matmul(fc1, tf.random_normal([16, n_classes]))
    out = tf.add(out, tf.random_normal([n_classes]))

    return out
