import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

class Line2d:
    def __init__(self, p0, p1):
        self.p0, self.p1 = p0, p1

        self.A = self.p1[1] - self.p0[1]
        self.B = self.p0[0] - self.p1[0]
        self.C = self.p0[0] * self.p1[1] \
               - self.p1[0] * self.p0[1]

    def __repr__(self):
        return "%s -- %s" % (self.p0, self.p1)

    def intersect(self, other):
        '''
        >>> l0 = Line2d((3,3),(3,4))
        >>> l1 = Line2d((1,1),(2,1))
        >>> l0.intersect(l1)
        False
        >>> l2 = Line2d((-3,-3), (-3,-4))
        >>> l3 = Line2d((-1,-1), (-2,-1))
        >>> l2.intersect(l3)
        False
        >>> l4 = Line2d((0,0), (1,1))
        >>> l5 = Line2d((0,1), (1,0))
        >>> l4.intersect(l5)
        True
        '''
        D  = self.A * other.B - self.B * other.A
        Dx = self.C * other.B - self.B * other.C
        Dy = self.A * other.C - self.C * other.A

        if D != 0.0:
            x = round(Dx / D, 6)
            y = round(Dy / D, 6)

            x_min = min(self.p0[0], self.p1[0])
            x_max = max(self.p0[0], self.p1[0])
            y_min = min(self.p0[1], self.p1[1])
            y_max = max(self.p0[1], self.p1[1])

            if x_min <= x and x <= x_max and \
                    y_min <= y and y <= y_max:
                return True
        return False

class Rectangle2d:
    def __init__(self, x_low, y_low, x_high, y_high):
        x_low, y_low   = round(x_low, 6),  round(y_low, 6)
        x_high, y_high = round(x_high, 6), round(y_high, 6)

        self.low = p0 = (x_low,y_low)
        p1 = (x_low,y_high)
        p2 = (x_high,y_low)
        self.high = p3 = (x_high,y_high)

        self.lines = [
            Line2d(p0,p1),
            Line2d(p0,p2),
            Line2d(p3,p1),
            Line2d(p3,p2)
        ]

    def intersect(self, other):
        '''
        #>>> r0 = Rectangle2d(0,0,1,1)
        #>>> r1 = Rectangle2d(2,2,3,3)
        #>>> r2 = Rectangle2d(1,1,2,2)
        #>>> r3 = Rectangle2d(1.5, 1.5, 2.5, 2.5)
        #>>> r0.intersect(r1)
        #False
        #>>> r0.intersect(r2)
        #True
        #>>> r1.intersect(r2)
        #True
        #>>> r1.intersect(r3)
        #True
        #>>> r2.intersect(r3)
        #True
        >>> r4 = Rectangle2d(0.64,0.56,0.99,0.68)
        >>> r5 = Rectangle2d(0.56,0.62,0.75,0.92)
        >>> r4.intersect(r5)
        True
        '''
        for line0 in self.lines:
            for line1 in other.lines:
                if line0.intersect(line1):
                    return True
        return False

    def inside(self, point):
        '''
        >>> r = Rectangle2d(0,0,1,1)
        >>> r.inside((0,0))
        True
        >>> r.inside((0.5,0.5))
        True
        >>> r.inside((2,2))
        False
        '''
        return self.low[0] <= point[0]     and \
               point[0]    <= self.high[0] and \
               self.low[1] <= point[1]     and \
               point[1]    <= self.high[1]

# DEPRECATED
def generate():

    rng = np.random.RandomState(42)
    random.seed(42)

    X0 = rng.uniform(low=-2, high=0, size=(400,2))
    y0 = [0.0 for _ in X0]

    X1 = rng.uniform(low=0, high=2, size=(400,2))
    y1 = [1.0 for _ in X1]

    X = np.append(X0, X1, axis = 0)
    y = y0 + y1

    X_out = np.append(
        np.append(
            rng.uniform(low=0,high=2, size=(400,1)),
            rng.uniform(low=-2,high=0,size=(400,1)),
            axis=1
        ),
        np.append(
            rng.uniform(low=-2,high=0, size=(400,1)),
            rng.uniform(low=0,high=2,size=(400,1)),
            axis=1
        ),
        axis = 0
    )
    y_out = [float(random.randint(0,1)) for _ in X_out]

    return np.append(X, X_out, axis=0), np.array(y + y_out)

def scatter2d(X,y):

    X0 = np.array([X[i] for i in range(len(X)) \
        if y[i] == 0.0])
    X1 = np.array([X[i] for i in range(len(X)) \
        if y[i] == 1.0])

    plt.scatter(X0[:,0], X0[:,1], c='red',s=1)
    plt.scatter(X1[:,0], X1[:,1], c='blue',s=1)
    plt.show()

def svc(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1)

    clf = SVC()

    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))

def generate_normalized_uniform_2d(
    size_out, size_ok, patches, acreage, seed = None
):
    rng = np.random.RandomState()
    if seed != None:
        rng = np.random.RandomState(seed)
        random.seed(seed)

    points = int(size_ok / patches)

    X = np.empty((0,2))
    y = []

    mem_patches = []
    for i in range(patches):
        # make sure patches do not overlap
        new_patch  = None
        while True:
            new_patch = rectangle(acreage)

            no_overlapping = True
            for patch in mem_patches:
                if patch.intersect(new_patch):
                    no_overlapping = False
                    break
            if no_overlapping:
                mem_patches.append(new_patch)
                break

        X = np.append(
            X,
            np.append(
                rng.uniform(
                    low  = new_patch.low[0],
                    high = new_patch.high[0],
                    size=(points,1)
                ),
                rng.uniform(
                    low  = new_patch.low[1],
                    high = new_patch.high[1],
                    size=(points,1)
                ),
                axis = 1,
            ),
            axis = 0
        )

        label = float(random.randint(0,1))
        y += [label for _ in range(points)]

    counter = 0
    while counter < size_out:
        while True:
            _0 = random.uniform(0,1)
            _1 = random.uniform(0,1)

            not_in_clean_patch = True
            for patch in mem_patches:
                if patch.inside((_0, _1)):
                    not_in_clean_patch = False
                    break
            if not_in_clean_patch:
                X = np.append(X, [[_0, _1]], axis = 0)
                y.append(float(random.randint(0,1)))
                break
        counter += 1

    return X, y

def rectangle(acreage):

    len_factor_x = random.random()
    len_factor_y = 1.0 - len_factor_x

    len_x = len_factor_x * (acreage / 2)
    len_y = len_factor_y * (acreage / 2)

    x_low  = random.uniform(0, 1.0 - len_x)
    x_high = x_low + len_x

    y_low  = random.uniform(0, 1.0 - len_y)
    y_high = y_low + len_y

    return Rectangle2d(round(x_low,6), round(y_low,6), round(x_high,6), round(y_high,6))

# TODO: abstractions, api (acreage, etc)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    X, y = generate_normalized_uniform_2d(3000,1500,2,1)
    scatter2d(X,y)
    #svc(X,y)
    #intersect()
