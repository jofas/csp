import numpy as np

from time import time

from sklearn.model_selection import train_test_split, KFold

from pcf import PartialClassificationForest
from plot import plot
from generate_dataset import generate_normalized_uniform_2d

def main():
    X, y = generate_normalized_uniform_2d(20000,0.2,5, 42)
    X, y = np.array(X), np.array(y)

    clf = PartialClassificationForest(
        n_estimators   = 20,
        min_leaf_size  = 5,
        gain_threshold = 0.99
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y)

    start = time()
    clf.fit(X_train, y_train)
    print('Fitting time: ', time() - start)

    score = clf.score(X_test, y_test)
    print(score)

    plot(clf, X, y)

def oy_main():
    from oy.main import meta, standardize, reduce_data
    from oy.import_data import import_data
    from oy.pca import pca

    from sklearn.decomposition import PCA

    X, y, _ = import_data('oy/data/clean.csv')
    print(len(X))
    X, y = reduce_data(X, y, 0.1)

    clf = PCA(n_components = 4)
    X = clf.fit_transform(X)

    print(clf.explained_variance_ratio_)
    print(sum(clf.explained_variance_ratio_))

    X, m = meta(X)
    X = standardize(X, m)

    y = [0.0 if x == -1.0 else 1.0 for x in y]
    X, y = np.array(X), np.array(y)

    print('Fitting...')
    clf = PartialClassificationForest(
        n_estimators   = 15,
        min_leaf_size  = 7,
        gain_threshold = 0.95
    )

    '''
    # train_test_split {{{
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.1)

    start = time()
    clf.fit(X_train, y_train)
    print('Fitting time:', time() - start)

    # how many % are predictable
    start = time()
    score = clf.score(X_train, y_train)
    print('Predicting time:', time() - start)
    print(score)

    # validation
    score = clf.score(X_test, y_test)
    print(score)
    # }}}
    '''

    # KFold {{{
    kf = KFold(n_splits = 5, shuffle = True)

    for train_i, test_i in kf.split(X):

        X_train, y_train = X[train_i], y[train_i]
        X_test,  y_test  = X[test_i],  y[test_i]

        start = time()
        clf.fit(X_train, y_train)
        print('Fitting time:', time() - start)

        # how many % are predictable
        start = time()
        score = clf.score(X_train, y_train)
        print('Predicting time:', time() - start)
        print(score)

        # validation
        score = clf.score(X_test, y_test)
        print(score)
    # }}}

    #plot(clf, X, y)

if __name__ == '__main__':
    oy_main()
    #main()
