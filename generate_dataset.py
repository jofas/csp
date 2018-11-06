import math
import numpy as np

from dimension2 import Rectangle2d, scatter2d

def generate_normalized_uniform_2d(
    points, ratio_clean, patches = -1, seed = None
):
    rng = np.random.RandomState(seed)
    patches = get_concrete_amount_of_patches(patches, rng)

    points_random, points_per_patch = \
        get_point_distribution(points,ratio_clean,patches)

    X = np.empty((0,2))
    y = []
    mem_patches = []
    for i in range(patches):
        mem_patches = append_new_patch(
            mem_patches, ratio_clean / patches, rng)

        X, y = append_points_in_patch(
            X, y, mem_patches[-1], points_per_patch, rng)

    X, y = append_points_outside_patches(
        X, y, mem_patches, points_random, rng)

    return X, y

def get_concrete_amount_of_patches(patches, rng):
    if type(patches) is range:
        return rng.randint(patches.start, patches.stop)
    elif patches == -1:
        return rng.randint(1,10)
    else:
        return patches

def get_point_distribution(points, ratio_clean, patches):
    return int(points * (1 - ratio_clean)), \
           int((points * ratio_clean) / patches)

def append_new_patch(patches, acreage, rng):
    while True:
        new_patch = rectangle(acreage, rng)
        if not overlap(new_patch, patches):
            patches.append(new_patch)
            return patches

def append_points_in_patch(X, y, patch, points, rng):
    X = np.append(X, np.append(rng.uniform(
                                    low  = patch.low[0],
                                    high = patch.high[0],
                                    size=(points,1)),
                                rng.uniform(
                                    low  = patch.low[1],
                                    high = patch.high[1],
                                    size=(points,1)),
                                axis = 1), axis = 0)

    label = float(rng.randint(0,2))
    y += [label for _ in range(points)]
    return X, y

def append_points_outside_patches(X,y,patches,points,rng):
    counter = 0
    while counter < points:
        X, y = append_new_point(X, y, patches, rng)
        counter += 1

    return X, y

def rectangle(acreage, rng):
    len_factor = rng.uniform(0.45,0.65)

    len_x = len_factor * math.sqrt(acreage)
    len_y = acreage / len_x

    x_low  = rng.uniform(0, 1.0 - len_x)
    x_high = x_low + len_x

    y_low  = rng.uniform(0, 1.0 - len_y)
    y_high = y_low + len_y

    return Rectangle2d(x_low, y_low, x_high, y_high)

def append_new_point(X, y, patches, rng):
    while True:
        x_ = rng.uniform(0,1)
        y_ = rng.uniform(0,1)

        if not inside_clean_patch(patches, x_, y_):
            return np.append(X, [[x_, y_]], axis = 0), \
                   y + [float(rng.randint(0,2))]

def overlap(new_rectangle, rectangles):
    for rectangle in rectangles:
        if new_rectangle.intersect(rectangle):
            return True
    return False

def inside_clean_patch(patches, x, y):
    for patch in patches:
        if patch.inside((x, y)):
            return True
    return False

if __name__ == '__main__':
    X, y = generate_normalized_uniform_2d(20000,0.20,5)
    scatter2d(X,y)
