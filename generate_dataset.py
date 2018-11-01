import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

class Point2d:
    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)

    def __repr__(self):
        return "(%s, %s)" % (self.x, self.y)

class Line2d:
    def __init__(self, p0, p1):
        self.p0, self.p1 = p0, p1
        self.A = p0.y - p1.y
        self.B = p1.x - p0.x
        self.C = p0.x * p1.y - p1.x * p0.y

    def __repr__(self):
        return "(%s) -- (%s)" % (self.p0, self.p1)

    def intersect(self, other):
        '''
        #>>> l0 = Line2d(Point2d(0,0),Point2d(0,1))
        #>>> l1 = Line2d(Point2d(1,0),Point2d(1,1))
        #>>> l2 = Line2d(Point2d(-0.5,0), Point2d(0.5,1.5))
        #>>> l0.intersect(l1)
        #False
        #>>> l0.intersect(l2)
        #True
        #>>> l1.intersect(l2)
        #True
        >>> l3 = Line2d(Point2d(3,3),Point2d(3,4))
        >>> l4 = Line2d(Point2d(1,1),Point2d(2,1))
        >>> l3.intersect(l4)
        False
        '''
        dx_self, dx_other  = \
            self.p0.x - self.p1.x, other.p0.x - other.p1.x
        dy_self, dy_other  = \
            self.p0.y - self.p1.y, other.p0.y - other.p1.y


        D = self.A * other.B  - self.B * other.A
        Dx = self.C * other.B - self.B * other.C
        Dy = self.A * other.C - self.C * other.A
        if self.A * other.B - self.B * other.A == 0:
            return False
        else:
            x = Dx / D
            y = Dy / D
            print(x)
            print(y)
            return True

class Rectangle2d:
    def __init__(self, x_low, y_low, x_high, y_high):
        p0 = Point2d(x_low,y_low)
        p1 = Point2d(x_low,y_high)
        p2 = Point2d(x_high,y_low)
        p3 = Point2d(x_high,y_high)

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
        #>>> r0.intersect(r1)
        #False
        #>>> r0.intersect(r2)
        #True
        #>>> r1.intersect(r2)
        #True
        '''
        for line0 in self.lines:
            for line1 in other.lines:
                print(line0)
                print(line1)
                if line0.intersect(line1):
                    return True
        return False

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

def generate_uniform_2d(
    lower, upper, size_out, size_ok, patches, patch_size,
    seed = None
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
        low_x  = None
        low_y  = None
        high_x = None
        high_y = None

        # make sure patches do not overlap
        while True:
            len_factor_x = random.random()
            len_factor_y = 1.0 - len_factor_x

            len_x = len_factor_x * patch_size
            len_y = len_factor_y * patch_size

            low_x = random.uniform(
                lower, upper - len_x)
            high_x = low_x + len_x

            low_y = random.uniform(
                lower, upper - len_y)
            high_y = low_y + len_y

            no_overlapping = True
            for lx, ly, hx, hy in mem_patches:
                if (
                    lx < low_x and low_x < hx and
                    ly < low_y and low_y < hy
                ) or (
                    lx < high_x and high_x < hx and
                    ly < high_y and high_y < hy
                ):
                    no_overlapping = False
                    break
            if no_overlapping:
                mem_patches.append((low_x,low_y,high_x,high_y))
                break

        X = np.append(
            X,
            np.append(
                rng.uniform(
                    low=low_x,
                    high=high_x,
                    size=(points,1)
                ),
                rng.uniform(
                    low=low_y,
                    high=high_y,
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
            x = random.uniform(lower,upper)
            z = random.uniform(lower,upper)

            not_in_clean_patch = True
            for lx, ly, hx, hy in mem_patches:
                if (
                    lx < x and x < hx
                ) and (
                    ly < z and z < hy
                ):
                    not_in_clean_patch = False
                    break
            if not_in_clean_patch:
                X = np.append(X, [[x, z]], axis = 0)
                y.append(float(random.randint(0,1)))
                break
        counter += 1

    return X, y




if __name__ == '__main__':
    import doctest
    doctest.testmod()
    #X, y = generate_uniform_2d(0,1,8000,200,10,0.1)
    #scatter2d(X,y)
    #svc(X,y)
    #intersect()
