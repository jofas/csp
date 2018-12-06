import unittest
import random
import numpy as np

def accuracy(y):
    nz = float(np.count_nonzero(y))
    ges = float(y.shape[0])
    return (0.0, 1.0 - nz / ges) if nz / ges < 0.5 else \
        (1.0, nz / ges)

class PartialClassifier:
    def __init__(
        self,
        n_estimators  = 5,
        min_leaf_size = 2,
        max_height    = 64,
        gain          = 'accuracy',
        threshold     = 0.8,
        splitter      = random.uniform,
    ):
        self.n_estimators  = n_estimators
        self.min_leaf_size = min_leaf_size
        self.max_height    = max_height
        self.gain          = \
            accuracy if gain == 'accuracy' else gain
        self.threshold     = threshold
        self.splitter      = splitter
        self.estimators    = \
            [None for _ in range(n_estimators)]

    def fit(self, X, y):

        boundries = np.array([(min(X[:,k]), max(X[:,k])) \
            for k in range(X.shape[1])])

        stack = [(X, y, boundries, 0)]

        for i in range(self.n_estimators):
            self.estimators[i] = self._estimator(stack[:])

    def _estimator(self, stack):

        tree = None

        while True:

            X, y, boundries, h = None, None, None, None

            try:
                X, y, boundries, h = stack.pop()
            except:
                return tree

            k = h % X.shape[1]

            label, gain_ = self.gain(y)

            if self.min_leaf_size > X.shape[0] or \
                    h == self.max_height:

                if tree == None:
                    tree = Leaf(-1.0)
                else:
                    tree.append(Leaf(-1.0), boundries)

            elif gain_ > self.threshold:

                if tree == None:
                    tree = Leaf(label)
                else:
                    tree.append(Leaf(label), boundries)

            else:

                split = self.splitter(
                    boundries[k,0], boundries[k,1])

                if tree == None:
                    tree = Node(split)
                else:
                    tree.append(Node(split), boundries)

                boundries_lower = np.copy(boundries)
                boundries_upper = np.copy(boundries)

                boundries_lower[k,1] = split
                boundries_upper[k,0] = split

                X_lower, y_lower = [], []
                X_upper, y_upper = [], []

                for i in range(X.shape[0]):
                    if X[i,k] <= split:
                        X_lower.append(X[i])
                        y_lower.append(y[i])
                    else:
                        X_upper.append(X[i])
                        y_upper.append(y[i])

                stack.append((
                    np.array(X_lower),
                    np.array(y_lower),
                    boundries_lower,
                    h + 1
                ))
                stack.append((
                    np.array(X_upper),
                    np.array(y_upper),
                    boundries_upper,
                    h + 1
                ))

    def score(self, X, y):
        pass

    def _V(self):
        v = 1.0
        for dim in self.meta:
            v *= dim[1] - dim[0]
        return v

class Node:
    def __init__(self, split):
        self.split = split
        self.left  = None
        self.right = None

    def append(self, NoL, boundries, __h = 0):
        if boundries[__h][0] < self.split:
            if self.left == None:
                self.left = NoL
            else:
                self.left.append(NoL, boundries,
                    (__h + 1) % len(boundries))
        else:
            if self.right == None:
                self.right = NoL
            else:
                self.right.append(NoL, boundries,
                    (__h + 1) % len(boundries))

class Leaf:
    def __init__(self, label):
        self.label = label

# TODO: test append (append via boundries),
#       accuracy fn + label_dict on stack, Leaf + label

class __TestPartialClassifier(unittest.TestCase):

    def test_tree_structure(self):
        def splitter(min, max):
            return float(min + max) / 2.0

        clf = PartialClassifier(
            n_estimators  = 1,
            min_leaf_size = 1,
            splitter      = splitter
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
        self.assertEqual(t.right.left.right.right.label,
            1.0)

    def test_max_height(self):
        pass

    def test_labels(self):
        pass

if __name__ == '__main__':
    unittest.main()
