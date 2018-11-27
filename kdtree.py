import numpy as np

class SymmetricKDTree:
    def __init__(self, X, y, axis = 0, split = 0.5,
            lor = None):

        self.split = split
        self.axis  = axis
        self.lor   = lor

        if self.is_leaf(X, y):
            self.X = X
            self.y = y
        else:
            self.left, self.right = \
                self.generate_childs(X, y)

    def generate_childs(self, X, y):
        lX, ly, rX, ry = [], [], [], []

        for i in range(X.shape[0]):
            if X[i][self.axis] <= self.split:
                lX.append(X[i])
                ly.append(y[i])
            else:
                rX.append(X[i])
                ry.append(y[i])

        l_split, r_split = None, None

        if self.axis == 0:
            l_split = r_split = self.split
        elif self.lor == 'left':
            l_split = r_split = self.split / 2
        else:
            l_split = r_split = self.split / 2 + self.split

        return SymmetricKDTree(
                np.array(lX),
                np.array(ly),
                (self.axis + 1) % X.shape[1],
                l_split,
                'left'
            ), SymmetricKDTree(
                np.array(rX),
                np.array(ry),
                (self.axis + 1) % X.shape[1],
                r_split,
                'right'
            )

    def is_leaf(self, X, y):
        return True if X.shape[0] <= 1 else False

    def __repr__(self, h = 0):
        padding = "\n" + str(self.axis) + " " + h * "  "

        if 'X' in self.__dict__:
            return padding + str(self.X)

        ret = padding + "{}".format(self.split)

        ret += self.left.__repr__(h+1)
        ret += self.right.__repr__(h+1)

        return ret



def main():
    X = [[0.0, 0.0],
         [0.0, 1.0],
         [1.0, 0.0],
         [1.0, 1.0]]

    y = [0.0, 0.0, 0.0, 1.0]

    t = SymmetricKDTree(np.array(X), np.array(y))

    print(t)

    pass

if __name__ == '__main__':
    main()
