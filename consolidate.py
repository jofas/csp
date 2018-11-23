import math
import unittest

import numpy as np

#from gmpy import popcount

class NeighborMatrix:
    def __init__(self, d, h):
        self.d = h if h < d else d
        self.offset = h - d if h > d else 0
        self.offset_matrix = self._offset_matrix()
        self.amnt = 0b1 << (self.d - 1)
        self.matrix = self._matrix()

    def _offset_matrix(self):
        m = np.array([[0]])
        for r in range(1, self.offset + 1):
            m_new = np.copy(m)

            for row in np.nditer(
                m_new, op_flags=['readwrite']
            ):
                row[...] = row + (0b1 << (r - 1))

            if r % 2 == 0:
                m = np.append(m, m_new, axis = 0)
            else:
                m = np.append(m, m_new, axis = 1)
        return m

    def _matrix(self):
        m = np.zeros((
            self.offset_matrix.shape[0] * 2,
            self.offset_matrix.shape[1] * self.amnt ))

        for i in range(self.offset_matrix.shape[0]):
            for j in range(m.shape[1]):
                offset = \
                    self.offset_matrix[i, int(j/self.amnt)]

                val = (j % self.amnt) ^ (offset << self.d)

                m[i*2][j]   = val
                m[i*2+1][j] = val ^ (0b1 << (self.d - 1))
        return m

def main():
    from time import time

    start = time()
    for i in range(1,7):
        print(i)
        nm = NeighborMatrix(2,i)
        print(nm.matrix)

    print("TIME: %f" % (time() - start))


if __name__ == '__main__':
    #unittest.main()
    main()



# Cube {{{
class Cube:
    def __init__(self, d):
        self.d = d
        self.layers = [[] for _ in range(self.d + 1)]
        self._compute_layers()

    def __repr__(self):
        ret = ''
        for k, layer in enumerate(self.layers):
            ret += '{}: ['.format(k)
            for val in layer:
                ret += '({:>{os}}, {:0>{dim}}), '.format(
                    val,
                    bin(val)[2:],
                    dim = self.d,
                    os  = math.ceil(math.log(2,10)*self.d)
                )
            ret = ret[:-2] + ']\n'

        return ret[:-1]

    def _compute_layers(self):
        for i in range(2 ** (self.d - 1)):
            k = popcount(i)
            self.layers[k].append(i)
            self.layers[self.d - k]\
                .append(self._complement(i))

    def _complement(self,num):
        return ((0b1 << self.d) - 0b1) ^ num
# }}}

class TestCube(unittest.TestCase):

    def flatten(self, l):
        new_list = []
        for i in l:
            if type(i) is list:
                new_list += self.flatten(i)
            else:
                new_list += [i]
        return new_list

    def test_flatten(self):
        x = [[0, 1], 2, 3, [4, [5, 6]]]
        self.assertEqual(
            self.flatten(x),
            [0, 1, 2, 3, 4, 5, 6]
        )

    def test_correctness(self):
        for i in range(1,18):
            cube = Cube(i)
            self.assertEqual(
                len(set(self.flatten(cube.layers))),
                2 ** i
            )

class Node:
    def __init__(self, d, value = 0):
        self.value = value

        if self.value == 2 ** d - 1:
            return

        self.next  = [None for _ in range(d)]

        for i in range(d):
            if (self.value >> i) % 2 == 0:
                self.next[i] = Node(
                    d, self.value + (1 << i))

    def __repr__(self, h = 0):

        padding = "\n" + h * "   "
        if 'next' not in self.__dict__:
            return padding + "{}".format(self.value)

        ret = padding + "{}:".format(self.value)

        for n in self.next:
            if n == None:
                ret += padding + "   Nil, "
            else:
                ret += "{} ".format(n.__repr__(h + 1))

        return ret

