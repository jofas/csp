import numpy as np

from consolidate import NeighborGraph

def discrete_threshold(y):
    if any(x != y[0] for x in y):
        return False
    return True

class Node:
    def __init__(self, tree, X, y, id = 0, axis = 0,
                 split = 0.5, lor = None, h = 0,
                 threshold = discrete_threshold,
                 min_leaf_size = 1):

        if h > tree.h:
            tree.h = h

        self.id    = id
        self.axis  = axis
        self.split = split
        self.lor   = lor
        self.h     = h

        if self.is_leaf(y, threshold, min_leaf_size):
            if X.shape[0] < min_leaf_size:
                self.label = -1
            else:
                self.amnt  = X.shape[0]
                self.ones  = np.count_nonzero(y)
                if float(self.ones) > self.amnt / 2:
                    self.label = 1
                else:
                    self.label = 0
        else:
            lX, ly, rX, ry   = self.split_data(X, y)
            l_split = r_split = self.new_split()

            self.left, self.right = Node(
                tree, np.array(lX), np.array(ly),
                self.id << 1, (self.axis + 1) % X.shape[1],
                l_split, 'left', self.h + 1, threshold,
                min_leaf_size
            ), Node(
                tree, np.array(rX), np.array(ry),
                (self.id << 1) ^ 0b1,
                (self.axis + 1) % X.shape[1], r_split,
                'right', self.h + 1, threshold,
                min_leaf_size
            )

    def predict(self, x):
        if 'label' in self.__dict__:
            return self.label
        elif x[self.axis] <= self.split:
            return self.left.predict(x)
        return self.right.predict(x)

    def get_leaf(self, id, max_h):
        if 'label' in self.__dict__:
            return self
        else:
            mask = 0b1 << (max_h - self.h - 1)
            if id & mask == mask:
                return self.right.get_leaf(id, max_h)
            return self.left.get_leaf(id,max_h)

    def is_leaf(self, y, threshold, min_leaf_size):
        if y.shape[0] <= min_leaf_size or threshold(y):
            return True
        return False

    def split_data(self, X, y):
        lX, ly, rX, ry = [], [], [], []
        for i in range(X.shape[0]):
            if X[i][self.axis] <= self.split:
                lX.append(X[i])
                ly.append(y[i])
            else:
                rX.append(X[i])
                ry.append(y[i])
        return lX, ly, rX, ry

    def new_split(self):
        if self.axis == 0:
            return self.split
        elif self.lor == 'left':
            return self.split / 2
        return self.split / 2 + self.split

    '''
    def __repr__(self, h = 0):
        padding = "\n" + str(self.axis) + " " + h * "  "

        if 'label' in self.__dict__:
            return padding+"{}: {}".format(
                self.id, self.label)

        ret = padding + "{}".format(self.id)

        ret += self.left.__repr__(h+1)
        ret += self.right.__repr__(h+1)

        return ret
    '''

class SymmetricKDTree:
    def __init__(self, X, y):
        self.h       = 0
        self.root    = Node(self, X, y)
        self.leaves  = [self.root.get_leaf(i, self.h) \
            for i in range(2 ** self.h)]
        self.graph   = NeighborGraph(X.shape[1], self.h)

        # ?? what to do with consolidation ???

    def predict(self, pred):
        return np.array(
            [float(self.root.predict(x)) for x in pred])

    '''
    def __repr__(self):
        return repr(self.root)
    '''

def plot(clf, X, y):
    import matplotlib.pyplot as plt
    import pylab as pl

    x_min, x_max = min(X[:,0]), max(X[:,0])
    y_min, y_max = min(X[:,1]), max(X[:,1])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.binary)#,alpha=0.5)
    '''
    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]
    '''

    l0 = np.array([X[i] for i in range(len(X)) if y[i] == 0.0])
    l1 = np.array([X[i] for i in range(len(X)) if y[i] == 1.0])

    plt.scatter(l0[:,0], l0[:,1], color = "b", s=1)
    plt.scatter(l1[:,0], l1[:,1], color = "r", s=1)
    plt.show()

def main():
    X = [[0.0, 0.0],
         [0.0, 1.0],
         [1.0, 0.0],
         [0.8, 0.3]]

    y = [0.0, 0.0, 0.0, 1.0]

    t = SymmetricKDTree(np.array(X), np.array(y))

    plot(t,X,y)

    pass

if __name__ == '__main__':
    main()
