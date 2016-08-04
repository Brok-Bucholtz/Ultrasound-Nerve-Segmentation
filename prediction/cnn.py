import tensorflow as tf
import math


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


def _filters_to_images(filters, image_count):
    images = []

    # Get the first <image_count> images of <filters>
    for filter_i in xrange(image_count):
        # Get dimensions
        filter_x = int(filters.get_shape()[1])
        filter_y = int(filters.get_shape()[2])
        channels = int(filters.get_shape()[3])
        channel_factor_pairs = [(i, channels / i) for i in range(1, int(channels**0.5)+1) if channels % i == 0]
        channel_y = channel_factor_pairs[len(channel_factor_pairs)-1][0]
        channel_x = channel_factor_pairs[len(channel_factor_pairs)-1][1]

        # Get ith filter
        image = tf.slice(filters, (filter_i, 0, 0, 0), (1, -1, -1, -1))
        image = tf.reshape(image, (filter_x, filter_y, channels))

        # Break up filter into its different channels for viewing
        filter_y += 4
        filter_x += 4
        image = tf.image.resize_image_with_crop_or_pad(image, filter_x, filter_y)
        image = tf.reshape(image, (filter_x, filter_y, channel_x, channel_y))
        image = tf.transpose(image, (2, 0, 3, 1))

        images.append(tf.reshape(image, (channel_x * filter_x, channel_y * filter_y, 1)))

    return images


def create_cnn(model_input, dropout, image_shape, n_classes):
    weights = {
        'conv1': tf.Variable(tf.random_normal([3, 3, 1, 16])),
        'conv2': tf.Variable(tf.random_normal([3, 3, 16, 32])),
        'conv3': tf.Variable(tf.random_normal([3, 3, 32, 64])),
        'conv4': tf.Variable(tf.random_normal([3, 3, 64, 128])),
        'conv5': tf.Variable(tf.random_normal([3, 3, 128, 256])),
        'conv6': tf.Variable(tf.random_normal([2, 2, 128, 256])),
        'conv7': tf.Variable(tf.random_normal([4, 4, 64, 128])),
        'conv8': tf.Variable(tf.random_normal([8, 8, 32, 64])),
        'conv9': tf.Variable(tf.random_normal([16, 16, 16, 32]))
    }
    biases = {
        'conv1': tf.Variable(tf.random_normal([16])),
        'conv2': tf.Variable(tf.random_normal([32])),
        'conv3': tf.Variable(tf.random_normal([64])),
        'conv4': tf.Variable(tf.random_normal([128])),
        'conv5': tf.Variable(tf.random_normal([256])),
        'conv6': tf.Variable(tf.random_normal([128])),
        'conv7': tf.Variable(tf.random_normal([64])),
        'conv8': tf.Variable(tf.random_normal([32])),
        'conv9': tf.Variable(tf.random_normal([16]))
    }

    model_input = tf.reshape(model_input, shape=[-1, image_shape[0], image_shape[1], 1])

    conv1 = _create_maxpool_layer(
        model_input,
        weights['conv1'],
        biases['conv1'])
    conv2 = _create_maxpool_layer(
        conv1,
        weights['conv2'],
        biases['conv2'])
    conv3 = _create_maxpool_layer(
        conv2,
        weights['conv3'],
        biases['conv3'])
    conv4 = _create_maxpool_layer(
        conv3,
        weights['conv4'],
        biases['conv4'])
    conv5 = _create_maxpool_layer(
        conv4,
        weights['conv5'],
        biases['conv5'])
    conv6 = _create_transpose_layer(
        conv5,
        [4, 4, 128],
        weights['conv6'],
        biases['conv6'])
    conv7 = _create_transpose_layer(
        conv6,
        [8, 8, 64],
        weights['conv7'],
        biases['conv7'])
    conv8 = _create_transpose_layer(
        conv7,
        [16, 16, 32],
        weights['conv8'],
        biases['conv8'])
    conv9 = _create_transpose_layer(
        conv8,
        [32, 32, 16],
        weights['conv9'],
        biases['conv9'])

    fc1 = tf.reshape(conv9, [-1, 32*32*16])
    fc1 = tf.matmul(fc1, tf.Variable(tf.random_normal([32*32*16, 16])))
    fc1 = tf.add(fc1, tf.Variable(tf.random_normal([16])))
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.matmul(fc1, tf.random_normal([16, n_classes]))
    out = tf.add(out, tf.random_normal([n_classes]))

    # Tensorboard
    tf.image_summary('Input', model_input)
    tf.image_summary('conv1', _filters_to_images(conv1, 3))
    tf.image_summary('conv5', _filters_to_images(conv5, 3))
    tf.image_summary('conv6', _filters_to_images(tf.reshape(conv6, [-1, 4, 4, 128]), 3))
    tf.image_summary('conv9', _filters_to_images(tf.reshape(conv9, [-1, 32, 32, 16]), 3))
    tf.image_summary('fc1', _filters_to_images(tf.reshape(fc1, [-1, 1, 1, 16]), 3))

    return out
