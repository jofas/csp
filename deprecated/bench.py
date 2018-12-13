import sys
from time import time
import numpy as np

def list_vs_numpy():
    l_start = time()
    l = [-1.0 for _ in range(10000)]
    l_end = time()

    n_start = time()
    n = np.full((10000,),-1.0)
    n_end = time()

    print("Initialized: \n\ttime: l: {}, n: {}\
                        \n\tmem:  l: {}, n: {}".format(
        l_end - l_start, n_end - n_start,
        sys.getsizeof(l), sys.getsizeof(n)
    ))

    l_start = time()
    l = []
    for i in range(10000):
        l.append(-1.0)
    l_end = time()

    n_start = time()
    n = np.array([])
    for i in range(10000):
        n = np.append(n, [-1.0], axis = 0)
    n_end = time()

    ln_start = time()
    ln = []
    for i in range(10000):
        ln.append(-1.0)
    ln = np.array(ln)
    ln_end = time()

    print("Append: \n\ttime: l: {}, n: {}, ln: {}\
                \n\tmem:  l: {}, n: {}, ln: {}".format(
        l_end - l_start,n_end - n_start,ln_end - ln_start,
        sys.getsizeof(l), sys.getsizeof(n),
        sys.getsizeof(ln)
    ))


if __name__ == '__main__':
    list_vs_numpy()
