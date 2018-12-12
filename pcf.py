import unittest
import random
import numpy as np

from multiprocessing import Pool

# TODO: - add support for new fitting
#       - test refitting
#       - remove all recursion (_Node.predict,
#           _Node.append)
#       - refactor _estimator
#           * Stack worker class with callback ??
#

# Processpool. Global, since otherwise it would be copied
# into the new processes which would lead to an
# PickleError.
_pool = None

def get_boundries(X):
    return np.array([(min(X[:,k]), max(X[:,k])) \
        for k in range(X.shape[1])])

class PartialClassificationForest:
    # def __init__ {{{
    def __init__(
        self,
        n_estimators   = 5,
        min_leaf_size  = 2,
        max_height     = 32,
        gain           = 'accuracy',
        gain_threshold = 0.8,
        splitter       = 'random',
    ):
        self.n_estimators   = n_estimators
        self.min_leaf_size  = min_leaf_size
        self.max_height     = max_height
        self.gain_threshold = gain_threshold
        self.splitter       = splitter
        self.estimators     = []
        self.gain           = \
            accuracy if gain == 'accuracy' else gain

        global _pool
        _pool = Pool()
    # }}}

    # def fit {{{
    def fit(self, X, y):
        label_count = LabelCount()
        for i in range(X.shape[0]):
            label_count.add(y[i])

        if -1.0 in label_count:
            raise Exception('-1.0 is an illegal label')

        if X.shape[0] < self.min_leaf_size:
            raise Exception('X must be greater or equal \
                to self.min_leaf_size')

        #with Pool() as pool:
        if len(self.estimators) == 0:

            boundries = get_boundries(X)
            #np.array([
            #    (min(X[:,k]), max(X[:,k])) \
            #        for k in range(X.shape[1])])

            res = [_pool.apply_async(self._estimator,
                (X, y, boundries, 0, label_count)) \
                    for _ in range(self.n_estimators)]
            self.estimators = [r.get() for r in res]
        else:
            res = [
                _pool.apply_async(self._refit_estimator,
                    (self.estimators[i], X, y,
                        label_count)) for i in range(
                            self.n_estimators)]
            self.estimators = [r.get() for r in res]
    # }}}

    # def _estimator {{{
    def _estimator(self, X, y, boundries, h, label_count):
        tree     = _Nil()
        stack    = [(X, y, boundries, h, label_count)]
        splitter = random.Random().uniform \
            if self.splitter == 'random' else self.splitter

        while True:
            X, y, boundries, h, label_count = \
                None, None, None, None, None

            try:
                X, y, boundries, h, label_count = \
                    stack.pop()
            except:
                return tree

            label, gain_ = self.gain(label_count,
                X.shape[0])

            # TODO: change append ?? possibly in stack,
            #       too -> take out every recursion
            if gain_ > self.gain_threshold:
                tree.append(_Leaf(label, X, y, boundries,
                    label_count), boundries)
            else:
                k    = h % X.shape[1]
                node = _Node(splitter(boundries[k,0],
                    boundries[k,1]))

                tree.append(node, boundries)

                boundries_low, boundries_up = \
                    node.split_boundries(boundries, k)

                X_low, X_up, y_low, y_up, \
                label_count_low, label_count_up = \
                    node.split_data(X, y, k)

                if self._tree_boundries(X_low, h + 1):
                    stack.append((X_low, y_low,
                        boundries_low, h + 1,
                        label_count_low))
                else:
                    node.left = _Leaf(-1.0, X_low, y_low,
                        boundries, label_count)

                if self._tree_boundries(X_up, h + 1):
                    stack.append((X_up, y_up, boundries_up,
                        h + 1, label_count_up))
                else:
                    node.right = _Leaf(-1.0, X_up, y_up,
                        boundries, label_count)
    # }}}

    # def _refit_estimator {{{
    def _refit_estimator(self,estimator,X, y, label_count):
        stack = [(estimator, X, y, 0, label_count)]
        splitter = random.Random().uniform \
            if self.splitter == 'random' else self.splitter

        while True:
            node, X, y, h, label_count = \
                None, None, None, None, None

            try:
                node, X, y, h, label_count = stack.pop()
            except:
                return estimator

            if type(node) is _Leaf:
                node.update(X, y, label_count)

                if self._tree_boundries(node.X, h):
                    new = self._estimator(node.X, node.y,
                        node.boundries, h,
                        node.label_count)

                    node.__class__ = new.__class__
                    node.__dict__  = new.__dict__
            else:
                k = h % X.shape[1]

                X_low, X_up, y_low, y_up, \
                label_count_low, label_count_up = \
                    node.split_data(X, y, k)

                if X_low.shape[0] > 0: stack.append((
                    node.left, X_low, y_low, h + 1,
                    label_count_low))
                if X_up.shape[0] > 0: stack.append((
                    node.right, X_up, y_up, h + 1,
                    label_count_up))
    # }}}

    # def _tree_boundries {{{
    def _tree_boundries(self, X, h):
        return X.shape[0] >= self.min_leaf_size and \
            h < self.max_height
    # }}}

    def predict(self, X):
        predictions = [LabelCount() for x in X]

        #with Pool() as pool:

        res = [_pool.apply_async(
            self.estimators[i].predict, (X,)) \
                for i in range(self.n_estimators)]

        for r in res:
            preds = r.get()
            for i in range(X.shape[0]):
                predictions[i].add(preds[i])

        return np.array([p.max()[0] for p in predictions])

    def score(self, X, y):
        labels = self.predict(X)

        unknown = 0
        correct = 0

        for i in range(X.shape[0]):
            if labels[i] == -1.0:
                unknown += 1
            elif y[i] == labels[i]:
                correct += 1
        try:
            return { 'known' : 1.0 - float(unknown) \
                             / float(X.shape[0]),
                     'acc'   : float(correct) \
                             / float(X.shape[0]-unknown) }
        except ZeroDivisionError:
            return {'known' : 0.0, 'acc' : 0.0 }

class _Node:
    def __init__(self, split = None):
        self.split = split
        self.left  = _Nil()
        self.right = _Nil()

    def append(self, NoL, boundries, __h = 0):
        if boundries[__h,0] < self.split:
            self.left.append(NoL, boundries,
                (__h + 1) % len(boundries))
        else:
            self.right.append(NoL, boundries,
                (__h + 1) % len(boundries))

    # TODO: in own stack
    def predict(self, X, __h = 0):
        X_low ,      X_up       = [], []
        X_low_index, X_up_index = [], []

        for i in range(X.shape[0]):
            if X[i,__h] <= self.split:
                X_low.append(X[i])
                X_low_index.append(i)
            else:
                X_up.append(X[i])
                X_up_index.append(i)

        res = [0 for _ in X]
        res = self.child_predict(res, self.left, X_low,
            X_low_index, __h)
        res = self.child_predict(res, self.right, X_up,
            X_up_index, __h)
        return res

    def child_predict(self, res, child, X, X_index, __h):
        if len(X) > 0:
            X = np.array(X)
            labels = child.predict(
                X, (__h + 1) % X.shape[1])
            for i in range(len(labels)):
                res[X_index[i]] = labels[i]
        return res

    def split_data(self, X, y, k):
        X_lower, y_lower = [], []
        X_upper, y_upper = [], []

        label_count_lower = LabelCount()
        label_count_upper = LabelCount()

        for i in range(X.shape[0]):
            if X[i,k] <= self.split:
                X_lower.append(X[i])
                y_lower.append(y[i])
                label_count_lower.add(y[i])
            else:
                X_upper.append(X[i])
                y_upper.append(y[i])
                label_count_upper.add(y[i])

        return np.array(X_lower), np.array(X_upper), \
               np.array(y_lower), np.array(y_upper), \
               label_count_lower, label_count_upper

    def split_boundries(self, boundries, k):
        boundries_lower = np.copy(boundries)
        boundries_upper = np.copy(boundries)

        boundries_lower[k,1] = self.split
        boundries_upper[k,0] = self.split

        return boundries_lower, boundries_upper

class _Leaf:
    def __init__(self,label, X, y, boundries, label_count):
        self.label       = label
        self.X           = X
        self.y           = y
        self.boundries   = boundries
        self.label_count = label_count

    def predict(self, X, _):
        return [self.label for _ in X]

    def update(self, X, y, label_count):
        self.X = np.append(self.X, X, axis = 0)
        self.y = np.append(self.y, y, axis = 0)
        self.label_count.append(label_count)
        X_boundries = get_boundries(X)
        for i in range(self.boundries.shape[0]):
            if X_boundries[i,0] < self.boundries[i,0]:
                self.boundries[i,0] = X_boundries[i,0]
            elif X_boundries[i,1] > self.boundries[i,1]:
                self.boundries[i,1] = X_boundries[i,1]

class _Nil:
    def append(self, NoL, _, __ = None):
        self.__class__ = NoL.__class__
        self.__dict__  = NoL.__dict__

class LabelCount:
    def __init__(self):
        self.count = {}

    def __contains__(self, key):
        return key in self.count

    def add(self, label, amnt = 1):
        if label not in self.count: self.count[label] = 0
        self.count[label] += amnt

    def append(self, other):
        for k, v in other.items(): self.add(k, v)

    def max(self):
        label, max = None, -1
        for k, v in self.items():
            if v > max: label, max = k, v
        return label, max

    def items(self):
        return self.count.items()

def accuracy(label_count, size):
    if size == 0:
        return -1, 0.0

    label, max = label_count.max()
    return label, float(max) / float(size)


###########################################################
#                                                         #
#                         TESTS                           #
#                                                         #
###########################################################


def _splitter_middle(min, max):
    return float(min + max) / 2.0

class __TestPCF(unittest.TestCase):
    def test_tree_structure(self):
        clf = PartialClassificationForest(
            n_estimators  = 1,
            min_leaf_size = 1,
            splitter      = _splitter_middle
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
        clf = PartialClassificationForest(
            n_estimators  = 1,
            min_leaf_size = 1,
            splitter      = _splitter_middle,
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
        clf = PartialClassificationForest(
            n_estimators   = 1,
            min_leaf_size  = 2,
            splitter       = _splitter_middle,
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

        labels = clf.predict(np.array([[0.3, 0.3],
                                       [0.3, 0.6],
                                       [0.6, 0.3],
                                       [0.6, 0.6]]))

        self.assertEqual(labels[0], 0.0)
        self.assertEqual(labels[1],-1.0)
        self.assertEqual(labels[2],-1.0)
        self.assertEqual(labels[3], 1.0)

    def test_score(self):
        clf = PartialClassificationForest(
            n_estimators   = 1,
            min_leaf_size  = 2,
            splitter       = _splitter_middle,
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
        clf = PartialClassificationForest(
            n_estimators   = 1,
            min_leaf_size  = 1,
            gain_threshold = 0.99,
        )

        X = np.array([[0.0, 0.0]])
        y = np.array( [0.0] )

        clf.fit(X, y)

        self.assertEqual(clf.estimators[0].label, 0.0)

    def test_refitting_split(self):
        clf = PartialClassificationForest(
            n_estimators   = 1,
            min_leaf_size  = 1,
            splitter       = _splitter_middle,
            gain_threshold = 0.99,
        )

        X = np.array([[0.0, 0.0]])
        y = np.array( [0.0] )

        clf.fit(X, y)
        self.assertEqual(clf.estimators[0].X.shape,X.shape)

        X = np.array([[1.0, 1.0]])
        y = np.array( [1.0] )

        t = clf.estimators[0]
        self.assertEqual(type(t), _Leaf)

        clf.fit(X, y)

        t = clf.estimators[0]


        self.assertEqual(t.split, 0.5)
        self.assertEqual(t.left.label, 0.0)
        self.assertEqual(t.right.label, 1.0)

    def test_refitting_wo_split(self):
        pass

if __name__ == '__main__':
    unittest.main()
