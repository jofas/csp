import numpy as np
import matplotlib.pyplot as plt

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

def scatter2d(X,y):
    X0 = np.array([X[i] for i in range(len(X)) \
        if y[i] == 0.0])
    X1 = np.array([X[i] for i in range(len(X)) \
        if y[i] == 1.0])

    plt.scatter(X0[:,0], X0[:,1], c='red',s=1)
    plt.scatter(X1[:,0], X1[:,1], c='blue',s=1)
    plt.show()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
