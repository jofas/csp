import random
import numpy as np

def accuracy(y):
    pass

class PartialClassifier:
    def __init__(
        self,
        n_estimators  = 5,
        min_leaf_size = 2,
        max_height    = 6,
        gain          = 'accuracy',
        threshold     = 0.8,
    ):
        self.n_estimators  = n_estimators
        self.min_leaf_size = min_leaf_size
        self.max_height    = max_height
        self.gain          = \
            accuracy if gain == 'accuracy' else gain
        self.threshold     = threshold

    def fit(self, X, y):

        n = X.shape[1]

        meta = np.array([(min(X[:,k]), max(X[:,k])) \
            for k in range(n)])

        stack = [(X,y)]

        # one estimator
        tree = Node()
        i = 0
        while i < self.max_height:
            X, y = stack.pop()

            if self.gain(y) > self.threshold:
                # append new leaf
                pass
            else:
                # split X and y and push them on the stack
                k = i % n
                split = random.uniform(meta[k,0],meta[k,1])

            i += 1

    def score(self, X, y):
        pass

class Node:
    def __init__(self, **kwargs):
        pass

    def to_leaf(self):
        pass

class Leaf:
    def __init__(self, **kwargs):
        pass

def main():
    X = np.array([ [0.0, 0.1, 0.2],
                   [1.0, 1.1, 1.2],
                   [2.0, 2.1, 2.2] ])

    y = np.array([ 0.3, 1.3, 2.3 ])

    clf = PartialClassifier()
    clf.fit(X, y)


if __name__ == '__main__':
    main()
