import unittest
import random
import numpy as np

def accuracy(label_count, size):
    if size == 0:
        return -1, 0.0

    label, max = label_count.max()
    return label, float(max) / float(size)

class LabelCount:
    def __init__(self):
        self.count = {}

    def __contains__(self, key):
        return key in self.count

    def add(self, label):
        if label not in self.count:
            self.count[label] = 0
        self.count[label] += 1

    def max(self):
        label, max = None, -1
        for k, v in self.items():
            if v > max: label, max = k, v
        return label, max

    def items(self):
        return self.count.items()

class PartialClassifier:
    def __init__(
        self,
        n_estimators   = 5,
        min_leaf_size  = 2,
        max_height     = 64,
        gain           = 'accuracy',
        gain_threshold = 0.8,
        splitter       = random.uniform,
    ):
        self.n_estimators   = n_estimators
        self.min_leaf_size  = min_leaf_size
        self.max_height     = max_height
        self.gain           = \
            accuracy if gain == 'accuracy' else gain
        self.gain_threshold = gain_threshold
        self.splitter       = splitter
        self.estimators     = \
            [None for _ in range(n_estimators)]

    def fit(self, X, y):

        boundries = np.array([(min(X[:,k]), max(X[:,k])) \
            for k in range(X.shape[1])])

        label_count = LabelCount()
        for i in range(X.shape[0]):
            label_count.add(y[i])

        if -1.0 in label_count:
            raise Exception('-1.0 is an illegal label')

        # TODO: add support for new fitting
        for i in range(self.n_estimators):
            self.estimators[i] = self._estimator(
                X, y, boundries, label_count)

    def _estimator(self, X, y, boundries, label_count):

        tree = Nil()
        stack = [(X, y, boundries, 0, label_count)]

        while True:

            X, y, boundries, h, label_count = \
                None, None, None, None, None

            try:
                X, y, boundries, h, label_count = \
                    stack.pop()
            except:
                return tree

            label, gain_ = self.gain(
                label_count, X.shape[0])

            if self.min_leaf_size > X.shape[0] or \
                    h == self.max_height:

                tree.append(Leaf(-1.0), boundries)

            elif gain_ > self.gain_threshold:

                tree.append(Leaf(label), boundries)

            else:

                k = h % X.shape[1]

                split = self.splitter(
                    boundries[k,0], boundries[k,1])

                tree.append(Node(split), boundries)

                boundries_lower = np.copy(boundries)
                boundries_upper = np.copy(boundries)

                boundries_lower[k,1] = split
                boundries_upper[k,0] = split

                X_lower, y_lower = [], []
                X_upper, y_upper = [], []

                label_count_lower = LabelCount()
                label_count_upper = LabelCount()

                for i in range(X.shape[0]):
                    if X[i,k] <= split:
                        X_lower.append(X[i])
                        y_lower.append(y[i])
                        label_count_lower.add(y[i])
                    else:
                        X_upper.append(X[i])
                        y_upper.append(y[i])
                        label_count_upper.add(y[i])

                stack.append((
                    np.array(X_lower),
                    np.array(y_lower),
                    boundries_lower,
                    h + 1,
                    label_count_lower
                ))
                stack.append((
                    np.array(X_upper),
                    np.array(y_upper),
                    boundries_upper,
                    h + 1,
                    label_count_upper
                ))

    def predict(self, X):
        ret = []
        for x in X:
            label_count = LabelCount()
            for estimator in self.estimators:
                label_count.add(estimator.predict(x))
            label, _ = label_count.max()
            ret.append(label)

        return np.array(ret)

    def score(self, X, y):
        labels = self.predict(X)

        unknown = 0
        correct = 0

        for i in range(X.shape[0]):
            if labels[i] == -1.0:
                unknown += 1
            elif y[i] == labels[i]:
                correct += 1

        return {
            'known' : \
                1.0-float(unknown)/float(X.shape[0]),
            'acc'  : \
                float(correct)/float(X.shape[0]-unknown),
        }

    '''
    def _V(self):
        v = 1.0
        for dim in self.meta:
            v *= dim[1] - dim[0]
        return v
    '''

class Node:
    def __init__(self, split = None):
        self.split = split
        self.left  = Nil()
        self.right = Nil()

    def append(self, NoL, boundries, __h = 0):
        if boundries[__h][0] < self.split:
            self.left.append(NoL, boundries,
                (__h + 1) % len(boundries))
        else:
            self.right.append(NoL, boundries,
                (__h + 1) % len(boundries))

    def predict(self, x, __h = 0):
        if x[__h] <= self.split:
            return self.left.predict(x, (__h + 1) % len(x))
        return self.right.predict(x, (__h + 1) % len(x))

class Leaf:
    def __init__(self, label):
        self.label = label

    def predict(self, _, __):
        return self.label

class Nil:
    def append(self, NoL, _, __ = None):
        self.__class__ = NoL.__class__
        self.__dict__  = NoL.__dict__

class __TestPartialClassifier(unittest.TestCase):

    def splitter_middle(self, min, max):
        return float(min + max) / 2.0

    def test_tree_structure(self):

        clf = PartialClassifier(
            n_estimators  = 1,
            min_leaf_size = 1,
            splitter      = self.splitter_middle
        )

        X = np.array([[0.0, 0.0],
                      [0.3, 0.0],
                      [0.6, 0.0],
                      [1.0, 0.0],
                      [0.0, 0.3],
                      [0.3, 0.3],
                      [0.6, 0.3],
                      [1.0, 0.3],
                      [0.0, 1.0],
                      [0.3, 1.0],
                      [1.0, 1.0]])
        y = np.array( [0.0,
                       0.0,
                       1.0,
                       0.0,
                       1.0,
                       1.0,
                       0.0,
                       1.0,
                       1.0,
                       0.0,
                       1.0] )

        clf.fit(X, y)

        t = clf.estimators[0]

        self.assertEqual(t.split, 0.5)

        self.assertEqual(t.left.split, 0.5)
        self.assertEqual(t.right.split, 0.5)

        self.assertEqual(t.left.left.split, 0.25)
        self.assertEqual(t.left.right.split, 0.25)
        self.assertEqual(t.right.left.split, 0.75)
        self.assertEqual(t.right.right.label, 1.0)

        self.assertEqual(t.left.left.left.split, 0.25)
        self.assertEqual(t.left.left.right.split, 0.25)
        self.assertEqual(t.left.right.left.label, 1.0)
        self.assertEqual(t.left.right.right.label, 0.0)
        self.assertEqual(t.right.left.left.split, 0.25)
        self.assertEqual(t.right.left.right.split, 0.25)

        self.assertEqual(t.left.left.left.left.label, 0.0)
        self.assertEqual(t.left.left.left.right.label, 1.0)
        self.assertEqual(t.left.left.right.left.label, 0.0)
        self.assertEqual(t.left.left.right.right.label,1.0)
        self.assertEqual(t.right.left.left.left.label, 1.0)
        self.assertEqual(t.right.left.left.right.label,0.0)
        self.assertEqual(t.right.left.right.left.label,0.0)
        self.assertEqual(
            t.right.left.right.right.label, 1.0)

    def test_max_height(self):
        clf = PartialClassifier(
            n_estimators  = 1,
            min_leaf_size = 1,
            splitter      = self.splitter_middle,
            max_height    = 3
        )

        X = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [1.0, 1.0]])
        y = np.array( [0.0,
                       1.0,
                       1.0] )

        clf.fit(X, y)

        t = clf.estimators[0]

        self.assertEqual(t.right.label, 1.0)
        self.assertEqual(t.left.right.label, -1.0)
        self.assertEqual(t.left.left.right.label, -1.0)
        self.assertEqual(t.left.left.left.label, -1.0)

    def test_predict(self):
        clf = PartialClassifier(
            n_estimators   = 1,
            min_leaf_size  = 2,
            splitter       = self.splitter_middle,
            gain_threshold = 0.99,
        )

        X = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0],
                      [1.0, 1.0]])

        y = np.array( [0.0,
                       0.0,
                       1.0,
                       0.0,
                       1.0,
                       1.0] )

        clf.fit(X, y)

        labels = clf.predict([[0.3, 0.3],
                              [0.3, 0.6],
                              [0.6, 0.3],
                              [0.6, 0.6]])

        self.assertEqual(labels[0], 0.0)
        self.assertEqual(labels[1],-1.0)
        self.assertEqual(labels[2],-1.0)
        self.assertEqual(labels[3], 1.0)

    def test_score(self):
        clf = PartialClassifier(
            n_estimators   = 1,
            min_leaf_size  = 2,
            splitter       = self.splitter_middle,
            gain_threshold = 0.99,
        )

        X = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0],
                      [1.0, 1.0]])

        y = np.array( [0.0,
                       0.0,
                       1.0,
                       0.0,
                       1.0,
                       1.0] )

        clf.fit(X, y)

        X_test = np.array([[0.3, 0.3],
                           [0.3, 0.6],
                           [0.6, 0.3],
                           [0.6, 0.6]])

        y_test = np.array( [0.0,
                            0.0,
                            0.0,
                            0.0] )

        score = clf.score(X_test, y_test)

        self.assertEqual(score['known'], 0.5)
        self.assertEqual(score['acc'], 0.5)

    def test_root_as_leaf(self):
        clf = PartialClassifier(
            n_estimators   = 1,
            min_leaf_size  = 1,
            gain_threshold = 0.99,
        )

        X = np.array([[0.0,0.0]])
        y = np.array([0.0])
        clf.fit(X, y)
        print(clf.estimators[0].__dict__)
        self.assertEqual(clf.estimators[0].label, 0.0)

if __name__ == '__main__':
    unittest.main()
