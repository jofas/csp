import random
import unittest
import math
import numpy as np
import matplotlib.pyplot as plt

# TODO: think first about the structure
#       a vector / plane interface (scipy/symbolic/knuth??)
#       generic (k)
#       scatter2d
#       tests
#       in rust + ffi ??

def scatter2d(kdtree):
    context = kdtree.context()
    points  = np.array(context['leafs'])

    plt.scatter(points[:,0],points[:,1])

    ''' TODO
    for index, split in enumerate(context['inner']):
        if split == None:
            continue
        if index == 0:
            plt.plot([(split,0),(split,3)])
        else:
            if int(math.log(index,2)) % 2 == 1:
                plt.plot([(split,0),(split,3)])
            else:
                plt.plot([(0,split),(3,split)])
    '''
    plt.show()

# Nil {{{
class Nil:
    def __init__(self, boundries = None):
        self.boundries = boundries
        self.axis = 1

    def direction(self, _):
        return 'right'

    def vectors(self,vector_list):
        return vector_list

    '''
    def vectors(self, axis, direction):
        if direction == 'right':

            pass
        elif direction == 'left':
            pass
    '''

    def get_split(self):
        return self.boundries[-1][0]

    def leafs(self):
        return []

    def __repr__(self):
        return "NIL({})".format(self.boundries)

    def __eq__(self,other):
        if other == None:
            return True
        if type(other) is Nil:
            return True
        return False
# }}}

class Node:

    nil = Nil()

    def __init__(
        self,k,points,axis,split,parent=None
    ):
        if parent == None:
            self.parent = Node.nil
        else:
            self.parent = parent

        self.split = next(split)
        self.axis  = axis
        self.k = k

        l = []
        r = []
        for point in points:
            if point[axis] <= self.split:
                l.append(point)
            else:
                r.append(point)

        self.left  = self._child(k,l,split)
        self.right = self._child(k,r,split)

    def __repr__(self):
        return "NODE(%s, %s)" % (self.axis, self.split)

    def get_split(self):
        return self.split

    def direction(self, child):
        return 'left' if self.left == child \
            else 'right'

    def vectors(self, vector_list=[]):
        direction = self.parent.direction(self)

        ps = self.parent.get_split()
        if self.axis == 0:
            self.a, self.b = (self.split, ps), (0,1)
        else:
            self.a, self.b = (ps, self.split), (1,0)

        up = self
        while up != Node.nil:
            up = up.parent
            if up.axis == self.axis:
                continue
            intersection = self.intersect(up)
            print(intersection)
            if (direction == 'left' and
                    intersection[self.axis] < self.split) \
            or (direction == 'right' and
                    intersection[self.axis] >= self.split):
                vector_list.append((self.a,intersection))
                break

        print(vector_list)
        vector_list = self.left.vectors(vector_list)
        vector_list = self.right.vectors(vector_list)
        return vector_list

    def intersect(self, other):
        alpha = None
        if other == Node.nil:
            a = other.boundries[(self.axis + 1) % self.k]
            b = tuple([1 if i == self.axis else 0 for i in range(self.k)])
            alpha = b[self.axis] - self.a[self.axis]
        else:
            alpha = other.b[self.axis] - self.a[self.axis]
        return [self.a[i] + alpha * self.b[i] \
            for i in range(len(self.a))]

    # _child {{{
    def _child(self, k, points, split):
        if len(points) > 1:
            return Node(k, points,(self.axis + 1) % k,
                split, self)
        elif len(points) == 1:
            return LeafNode(points[0])
        else:
            return Node.nil
    # }}}

    # splits {{{
    #
    # breadth first
    def splits(self):
        queue = [self]
        splits = []

        while queue != []:
            node = queue.pop(0)
            if type(node) is Nil:
                splits.append(None)
            elif type(node) is Node:
                splits.append(node.split)
                queue.append(node.left)
                queue.append(node.right)
        return splits
    # }}}

    # leafs {{{
    def leafs(self):
        return self.left.leafs() + self.right.leafs()
    # }}}

# LeafNode {{{
class LeafNode:
    def __init__(self, point):
        self.point = point

    def __repr__(self):
        return "LEAFNODE({})".format(self.point)

    def vectors(self, vector_list):
        return vector_list

    def leafs(self):
        return [self.point]
# }}}

# KDTree {{{
class KDTree:
    def __init__(self, k, points, split = random.random):
        self.k = k

        boundries = []
        for i in range(k):
            col = [x[i] for x in points]
            boundries.append((min(col),max(col)))

        print(Node.nil)
        Node.nil = Nil(boundries)
        print(Node.nil)
        self.tree = Node(
            k, points, 0, split)

    def context(self):
        return {
            "k":     self.k,
            "inner": self.tree.splits(),
            "leafs": self.tree.leafs(),
        }
# }}}

class __TestKDTree(unittest.TestCase):

    def test_1(self):
        print('\n--- TEST 1 ---')
        def split1():
            split = [1, 1, 1]
            for i in split:
                yield 1

        kd = KDTree(2, [(0,0),(2,0),(0,2),(2,2)], split1())

        self.assertEqual(kd.tree.left.left.point, (0,0))
        self.assertEqual(kd.tree.left.right.point, (0,2))
        self.assertEqual(kd.tree.right.left.point, (2,0))
        self.assertEqual(kd.tree.right.right.point, (2,2))

        kd_context = kd.context()

        self.assertEqual(kd_context['k'], 2)
        self.assertEqual(kd_context['inner'], [1, 1, 1])
        self.assertEqual(kd_context['leafs'],
            [(0,0),(0,2),(2,0),(2,2)])

        vectors = kd.tree.vectors()
        print(vectors)

if __name__ == '__main__':
    unittest.main()

    def split():
        split = [3, 3, 1, 1, 1]
        for i in range(len(split)):
            yield split[i]

    scatter2d(KDTree(2,[(0,0),(2,0),(0,2),(2,2)],split()))
