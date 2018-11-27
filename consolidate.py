#import math
import unittest
import sys
#import numpy as np

from copy import deepcopy
from gmpy import popcount

class NeighborGraph:
    def __init__(self, d, h):
        self.d      = h if h < d else d
        self.offset = h - d if h > d else 0

        self.graph = _Fold(self.d)

        for o in range(self.offset):
            graph_ = None
            while True:
                try:
                    graph_ = deepcopy(self.graph)
                    break
                except RecursionError:
                    sys.setrecursionlimit(
                        sys.getrecursionlimit() * 10)

            for i in range(len(graph_.nodes)):
                graph_.nodes[i].id = graph_.nodes[i].id \
                    + (0b1 << (o + self.d))

            self.graph = self.graph.append(graph_)

    def conns(self):
        return self.graph.nodes[0].conns()

class _Fold:
    def __init__(self, n):
        self.n         = n
        self.sides     = [[] for _ in range(n)]
        self.sides_    = [[] for _ in range(n)]
        self.nodes     = []
        self.rotation  = 0
        self.direction = 1

        for i in range(2 ** n):

            x = _DirectedNode(i)
            for node in self.nodes:
                if popcount(x.id ^ node.id) == 1:
                    node.edges_out.append(x)
            self.nodes.append(x)

            for j in range(n):
                mask = 0b1 << j
                if x.id & mask == mask:
                    self.sides[j].append(x)
                else:
                    self.sides_[j].append(x)

    def append(self, other):
        for i in range(len(self.sides[self.rotation])):
            self.sides[self.rotation][i].edges_out.append(
                other.sides_[self.rotation][i])

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

        self.nodes += other.nodes

        return self

class _DirectedNode:
    def __init__(self, id):
        self.id        = id
        self.edges_out = []

    def __repr__(self):
        return "DCN({})".format(self.id)

    def conns(self, h = 0):
        padding = "\n" + h * "  "
        ret = padding + "{}".format(self.id)

        for node in self.edges_out:
            ret += node.conns(h+1)
        return ret

def main():
    ng = NeighborGraph(4,4)
    print(ng.conns())

if __name__ == '__main__':
    #unittest.main()
    main()
















class NeighborMatrix:
    def __init__(self, d, h):
        self.d             = h if h < d else d
        self.offset        = h - d if h > d else 0
        self.offset_matrix = self._offset_matrix()
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

