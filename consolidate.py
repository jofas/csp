import math
import unittest

import numpy as np

from copy import deepcopy
from gmpy import popcount


class NeighborMatrix:
    def __init__(self, d, h):
        self.d             = h if h < d else d
        self.offset        = h - d if h > d else 0
        #self.amnt          = 0b1 << (self.d - 1)
        self.offset_matrix = self._offset_matrix()
        #self.matrix        = self._matrix()
        self.group = self._group()

    def _offset_matrix(self):
        m = np.array([0], ndmin = self.d)
        for r in range(1, self.offset + 1):
            m_new = np.copy(m)

            for row in np.nditer(
                m_new, op_flags = ['readwrite']
            ):
                row[...] = row + (0b1 << (r - 1))

            m = np.append(m, m_new, axis = r % self.d)

        return m

    def _group(self):
        m = np.array([0], ndmin = self.d)
        for r in range(1, self.d + 1):
            m_new = np.copy(m)

        return m
        #for r in range(1, 2 ** self.d):


    ''' {{{
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
    }}} '''

class DirectedCubeNode:
    def __init__(self, id, n):
        self.id        = id
        self.n         = n
        self.edges_in  = []
        self.edges_out = []
        #self.edges = []

    '''
    def connect(self, other):
        if popcount(self.id ^ other.id) == 1:
            self.edges.append(other)
        else:
            for edge in self.edges:
                edge.connect(other)
    '''


    def __repr__(self, h = 0):

        return "DCN({})".format(self.id)

        padding = "\n" + h * "   "
        ret = padding + "{}".format(self.id)

        for x in self.edges:
            ret += "{} ".format(x.__repr__(h + 1))

        return "{}: {}".format(self.id, self.edges)

    def conns(self):
        ret = "{}: [".format(self.id)
        for node in self.edges_out:
            ret += node.conns() + ", "
        return ret[:-2] + "]" if len(self.edges_out) > 0 \
            else ret[:-3]


class Fold:
    def __init__(self, n):
        self.n        = n
        self.sides     = [[] for _ in range(n)]
        self.sides_    = [[] for _ in range(n)]
        self.nodes     = []
        self.rotation  = 0
        self.direction = 1

        for i in range(2 ** n):

            x = DirectedCubeNode(i,n)
            for node in self.nodes:
                if popcount(x.id ^ node.id) == 1:
                    x.edges_in.append(node)
                    node.edges_out.append(x)
            self.nodes.append(x)

            for j in range(n):
                mask = 0b1 << j
                if x.id & mask == mask:
                    self.sides[j].append(x)
                else:
                    self.sides_[j].append(x)

    def conns(self):
        return self.nodes[0].conns()

    def _append(self, other):
        # connect every node at self.rotation from self with
        # the complement of the side at self.rotation of other
        for i in range(len(self.sides[self.rotation])):
            self.sides[self.rotation][i].edges_out.append(
                other.sides_[self.rotation][i])
            other.sides_[self.rotation][i].edges_in.append(
                self.sides[self.rotation][i])

        '''
        side  = self.sides.pop(0)
        side_ = other.sides_.pop(0)
        for i in range(len(side)):
            side[i].edges_out.append(side_[i])
            side_[i].edges_in.append(side[i])

        new_side  = other.sides.pop(0)
        new_side_ = self.sides_.pop(0)

        for i in range(len(self.sides)):
            self.sides[i]  += other.sides[i]
            self.sides_[i] += other.sides_[i]

        self.sides.append(new_side)
        self.sides_.append(new_side_)
        '''
        # add the nodes of other to the nodes of self
        self.nodes += other.nodes

        for i in range(self.n):
            if i == self.rotation:
                self.sides[i] = other.sides[i]
            else:
                self.sides[i] += other.sides[i]
                self.sides_[i] += other.sides_[i]

        self.rotation += self.direction

        if self.rotation == self.n:
            self.direction = -1
            self.rotation += self.direction
        elif self.rotation == -1:
            self.direction = 1
            self.rotation += self.direction

        return self

    '''
    def copy_with_prefix(self, prefix):
        new_cube = deepcopy(self)
        for i in range(len(new_cube.nodes)):
            new_cube.nodes[i].id = new_cube.nodes[i].id \
                + (prefix << self.n)
        return new_cube
    '''

def main():
    bits_offset = 1

    x = Fold(3)

    for offset in range(bits_offset):
        print("OFFSET: ", offset)
        #print("NODES: ", x.nodes)
        print("SIDES: ", x.sides, "\n")
        #print("CONNS: ", x.conns(), "\n")

        new_fold = deepcopy(x)
        for i in range(len(new_fold.nodes)):
            new_fold.nodes[i].id = new_fold.nodes[i].id \
                + (0b1 << (offset + x.n))

        x = x._append(new_fold)

    print("FINAL")
    #print("NODES: ", x.nodes)
    print("SIDES: ", x.sides)
    #print("CONNS: ", x.conns(), "\n")


    '''
    from time import time

    start = time()
    for i in range(1,8):
        print(i)
        nm = NeighborMatrix(2,i)
        print(nm.offset_matrix)

    print("TIME: %f" % (time() - start))
    '''

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

