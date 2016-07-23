from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from feature_extraction import get_detection_data


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


if __name__ == '__main__':
    # KNN Detection Model
    # F1 score for training set: 0.997677119628
    # F1 score for test set: 0.696686491079
    _run_knn_detection()

    # SVM Detection Model
    # F1 score for training set: 0.770956684325
    # F1 score for test set: 0.682745825603
    _run_svm_detection()
