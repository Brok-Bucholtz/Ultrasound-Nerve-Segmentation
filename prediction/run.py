import tensorflow as tf
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

from feature_extraction import get_detection_data
from prediction.cnn import create_cnn


def _plot_learning_curve(model, features, labels, title='', scoring=None):
    last_backend = plt.get_backend()
    plt.switch_backend('Agg')
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(model, features, labels, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r")
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g")
    plt.plot(
        train_sizes,
        train_scores_mean,
        'o-',
        color="r",
        label="Training score")
    plt.plot(
        train_sizes,
        test_scores_mean,
        'o-',
        color="g",
        label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig(title + '.png')
    plt.switch_backend(last_backend)


def _run_knn_detection(x_train, x_test, y_train, y_test, accuracy_function):
    clf = KNeighborsClassifier(2, 'distance')

    print "Training KNN..."
    _plot_learning_curve(clf, x_train, y_train, 'KNN-Learning-Curve', make_scorer(accuracy_function))
    clf.fit(x_train, y_train)
    print "Predicting Test Set..."
    print "Score for test set: {}".format(accuracy_function(y_test, clf.predict(x_test)))


def _run_svm_detection(x_train, x_test, y_train, y_test, accuracy_function):
    clf = SVC(C=9)

    print "Training SVM..."
    _plot_learning_curve(clf, x_train, y_train, 'SVM-Learning-Curve', make_scorer(accuracy_function))
    clf.fit(x_train, y_train)
    print "Predicting Test Set..."
    print "Score for test set: {}".format(accuracy_function(y_test, clf.predict(x_test)))


def _run_cnn_detection(x_train, x_test, y_train, y_test, accuracy_function):
    SUMMARY_PATH = '/tmp/ultrasound-never-segmentation/summary'
    learning_rate = 0.001
    batch_size = 161
    image_shape = (42, 58)
    n_classes = 2
    keep_prob = 0.75

    model_input = tf.placeholder(tf.float32, [None, image_shape[0]*image_shape[1]])
    model_output = tf.placeholder(tf.float32, [None, n_classes])
    dropout = tf.placeholder(tf.float32)

    y_train = [[float(y_element), float(not y_element)] for y_element in y_train]
    y_test = [[float(y_element), float(not y_element)] for y_element in y_test]

    cnn_model = create_cnn(model_input, dropout, image_shape, n_classes)

    prediction = tf.argmax(cnn_model, 1)
    cost = tf.nn.softmax_cross_entropy_with_logits(cnn_model, model_output)
    cost = tf.reduce_mean(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    tf.scalar_summary('dropout', dropout)
    correct_prediction = tf.equal(tf.argmax(cnn_model, 1), tf.argmax(model_output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

    with tf.Session() as sess:
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(SUMMARY_PATH + '/train',sess.graph)
        sess.run(tf.initialize_all_variables())

        print "Training CNN..."
        for batch_i in xrange(int(ceil(len(x_train)/float(batch_size)))):
            front_batch = batch_i * batch_size
            batch_x = x_train[front_batch: front_batch + batch_size]
            batch_y = y_train[front_batch: front_batch + batch_size]

            summary, _ = sess.run([merged, optimizer], feed_dict={
                model_input: batch_x,
                model_output: batch_y,
                dropout: keep_prob})
            train_writer.add_summary(summary, batch_i)

        print "F1 score for train set: {}".format(accuracy_function(
            [y[1] for y in y_train],
            [sess.run(prediction, feed_dict={model_input: [input], dropout: 1.0})[0] for input in x_train]))
        print "F1 score for test set: {}".format(accuracy_function(
            [y[1] for y in y_test],
            [sess.run(prediction, feed_dict={model_input: [input], dropout: 1.0})[0] for input in x_test]))


def run():
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(*get_detection_data(), test_size=0.25)
    accuracy_function = f1_score
    # KNN Detection Model
    # F1 score for training set: 0.997677119628
    # F1 score for test set: 0.696686491079
    _run_knn_detection(x_train, x_test, y_train, y_test, accuracy_function)

    # SVM Detection Model
    # F1 score for training set: 0.770956684325
    # F1 score for test set: 0.682745825603
    _run_svm_detection(x_train, x_test, y_train, y_test, accuracy_function)

    # CNN Detection Model
    # F1 score for training set: 0.535408901557
    # F1 score for test set: 0.577373211964
    _run_cnn_detection(x_train, x_test, y_train, y_test, accuracy_function)


if __name__ == '__main__':
    run()
