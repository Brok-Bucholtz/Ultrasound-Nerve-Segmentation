import tensorflow as tf
import cv2
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from math import ceil

from feature_extraction import get_detection_data
from prediction.cnn import create_cnn, get_predictions


def _run_knn_detection():
    x_all, y_all = get_detection_data()
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_all, y_all, test_size=0.25)

    clf = KNeighborsClassifier(2, 'distance')
    print "Training KNN..."
    clf.fit(x_train, y_train)
    print "Predicting Training Set..."
    print "F1 score for training set: {}".format(f1_score(y_train, clf.predict(x_train)))
    print "Predicting Test Set..."
    print "F1 score for test set: {}".format(f1_score(y_test, clf.predict(x_test)))


def _run_svm_detection():
    x_all, y_all = get_detection_data()
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_all, y_all, test_size=0.25)
    clf = SVC(C=9)

    print "Training SVM..."
    clf.fit(x_train, y_train)
    print "Predicting Training Set..."
    print "F1 score for training set: {}".format(f1_score(y_train, clf.predict(x_train)))
    print "Predicting Test Set..."
    print "F1 score for test set: {}".format(f1_score(y_test, clf.predict(x_test)))


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

    prediction, optimizer = create_cnn(model_input, model_output, dropout, (image_width, image_height), n_classes)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        print "Training CNN..."
        for batch_i in xrange(int(ceil(len(x_train)/float(batch_size)))):
            front_batch = batch_i * batch_size
            batch_x = x_train[front_batch: front_batch + batch_size]
            batch_y = y_train[front_batch: front_batch + batch_size]

            sess.run(optimizer, feed_dict={model_input: batch_x, model_output: batch_y, keep_prob: dropout})

        print "F1 score for train set: {}".format(f1_score(
            [y[1] for y in y_train],
            get_predictions(sess, prediction, model_input, x_train)))
        print "F1 score for test set: {}".format(f1_score(
            [y[1] for y in y_test],
            get_predictions(sess, prediction, model_input, x_test)))


if __name__ == '__main__':
    # KNN Detection Model
    # F1 score for training set: 0.997677119628
    # F1 score for test set: 0.696686491079
    _run_knn_detection()

    # SVM Detection Model
    # F1 score for training set: 0.770956684325
    # F1 score for test set: 0.682745825603
    _run_svm_detection()

    # CNN Detection Model
    # F1 score for training set: 0.535408901557
    # F1 score for test set: 0.577373211964
    run_cnn_detection()
