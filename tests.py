import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from pcf import PartialClassificationForest
from gen_2d import generate_normalized_uniform_2d

def export_csv(path, data):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for row in data:
            writer.writerow(row)

def export_dataset_mesh(clf, X, y, path):
    x_min, x_max = min(X[:,0]), max(X[:,0])
    y_min, y_max = min(X[:,1]), max(X[:,1])

    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h))

    X_mesh = np.c_[xx.ravel(), yy.ravel()]

    Z = clf.predict(X_mesh)

    mesh_exp_0 = []
    mesh_exp_1 = []
    data_exp_0 = []
    data_exp_1 = []

    for i in range(X_mesh.shape[0]):
        if Z[i] == 0.0:
            mesh_exp_0.append(X_mesh[i])
        elif Z[i] == 1.0:
            mesh_exp_1.append(X_mesh[i])

    for i in range(X.shape[0]):
        if y[i] == 0.0:
            data_exp_0.append(X[i])
        elif y[i] == 1.0:
            data_exp_1.append(X[i])

    export_csv(path + '.mesh_0.csv', mesh_exp_0)
    export_csv(path + '.mesh_1.csv', mesh_exp_1)
    export_csv(path + '.data_0.csv', data_exp_0)
    export_csv(path + '.data_1.csv', data_exp_1)

def export_score(score, path):
    with open(path + ".tex", "w") as file:
        file.write(score)

def test_other_classifiers():

    X, y = generate_normalized_uniform_2d(5000,0.2,5, 42)
    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.1)

    clf = SVC()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    export_dataset_mesh(clf, X, y, "svm")
    export_score(str(score), "svm_acc")
    print("SVM: ", score)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    export_dataset_mesh(clf, X, y, "rf")
    export_score(str(score), "rf_acc")
    print("RF: ", score)

    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    export_dataset_mesh(clf, X, y, "nn")
    export_score(str(score), "nn_acc")
    print("NN: ", score)

def test_descision_surface(ns, ms):
    X, y = generate_normalized_uniform_2d(5000,0.2,5, 42)
    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.1)

    for n_estimators in ns:
        for min_leaf_size in ms:
            print("\n{}.{}".format(n_estimators,
                min_leaf_size))
            clf = PartialClassificationForest(
                n_estimators   = n_estimators,
                min_leaf_size  = min_leaf_size,
                gain_threshold = 0.99
            )

            clf.fit(X_train, y_train)

            # how many % are predictable
            score = clf.score(X_train, y_train)
            print(score)

            # validation
            score = clf.score(X_test, y_test)
            print(score)

            path = "e{}_ls{}".format(n_estimators,
                min_leaf_size)
            #plot_descision_surface(clf, X, y, path)
            export_dataset_mesh(clf, X, y, path)
            export_score(str(score['known']),
                path + '_pred')
            export_score(str(score['acc']),
                path + '_acc')

def test_estimator_size():
    X, y = generate_normalized_uniform_2d(5000,0.2,5, 42)
    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.1)

    known_fit, known_pred, acc_fit, acc_pred = [],[],[],[]

    for i in [150,200]:
        print("\n{}".format(i))

        clf = PartialClassificationForest(
            n_estimators   = i,
            min_leaf_size  = 4,
            gain_threshold = 0.99
        )

        clf.fit(X_train, y_train)

        # how many % are predictable
        score = clf.score(X_train, y_train)
        print(score)

        known_fit.append([i, score['known']])
        acc_fit.append([i, score['acc']])

        # validation
        score = clf.score(X_test, y_test)
        print(score)

        known_pred.append([i, score['known']])
        acc_pred.append([i, score['acc']])

        del clf

    export_csv("fit_known.csv", known_fit)
    export_csv("fit_acc.csv", acc_fit)
    export_csv("pred_known.csv", known_pred)
    export_csv("pred_acc.csv", acc_pred)

if __name__ == '__main__':
    test_descision_surface([2, 5, 10], [2, 5, 10])
    #test_descision_surface([2], [2])
    #test_estimator_size()
    #test_other_classifiers()

# {{{
def plot_descision_surface(clf, X, y, path):
    plt.figure(figsize=(5,5))

    x_min, x_max = min(X[:,0]), max(X[:,0])
    y_min, y_max = min(X[:,1]), max(X[:,1])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    print(xx.shape)
    print(yy.shape)
    print(Z.shape)
    return

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.binary)#,alpha=0.5)

    l0 = np.array([X[i] for i in range(len(X)) if y[i] == 0.0])
    l1 = np.array([X[i] for i in range(len(X)) if y[i] == 1.0])

    plt.scatter(l0[:,0], l0[:,1], color = "b", s=0.05)
    plt.scatter(l1[:,0], l1[:,1], color = "r", s=0.05)
    plt.savefig(path)
    #plt.show()
    plt.clf()
# }}}

