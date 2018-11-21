import math
import unittest

from gmpy import popcount

def connection_graph(d,h):
    if h < d:
        return connection_graph(h,h)
    elif h == d:
        g = Cube(d)
        print(g)
    if h > d:
        gs = 2 ** (h - d)
        return None

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

def main():
    connection_graph(5,5)

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

if __name__ == '__main__':
    unittest.main()
    main()




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

