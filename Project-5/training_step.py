import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lesson_functions import *

import numpy as np
import cv2
import glob
import time
import pickle

threshold = 0

if __name__ == '__main__':
    # load data
    target_names = ['No car', 'car']

    data = np.load('data.npy')
    labels = np.load('labels.npy')
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)

    clf = LinearSVC(max_iter=20000, loss='hinge')#XGBClassifier(n_estimators=25, learning_rate=0.5, nthread=8, objective='binary:logistic')
    clf.fit(X_train, y_train)
    print("training complete.")
    # plot
    #plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
    #plt.show()

    #if threshold == 0:
    #    index = [elem for elem in clf.feature_importances_ if elem > threshold]
    #    pickle.dump(index, open('best_index.p', 'wb'))
    #    print(len(index))


    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=target_names))
    pickle.dump(clf, open("classifier.p", "wb"))

