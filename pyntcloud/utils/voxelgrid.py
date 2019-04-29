import itertools
import math
import numpy as np

import itertools
import math

np.seterr(divide='ignore', invalid='ignore')

def get_all_combinations(lst):
    combs = []
    for i in range(1, len(lst)):
        els = [list(x) for x in itertools.combinations(lst, i)]
        combs.extend(els)
    return combs

def supercover_line(p0, p1, resolution):
    assert len(p0) == len(p1) == len(resolution)
    lengths = np.abs(p1 - p0) / resolution
    steps = 1 / lengths
    pos = np.floor(p0 / resolution)
    final_pos = np.floor(p1 / resolution)
    t_nexts = np.zeros(p0.shape)
    increments = np.zeros(p0.shape)
    for dim in range(0, len(p0)):
        if lengths[dim] == 0:
            t_nexts[dim] = np.inf
        elif p1[dim] > p0[dim]:
            increments[dim] = 1
            next_ = math.floor(p0[dim] / resolution[dim]) * resolution[dim] + resolution[dim]
            t_nexts[dim] = ((next_ - p0[dim]) / resolution[dim]) * steps[dim]
        else:
            increments[dim] = -1
            next_ = math.floor(p0[dim] / resolution[dim]) * resolution[dim]
            t_nexts[dim] = ((p0[dim] - next_) / resolution[dim]) * steps[dim]
    yield pos.copy()
    while True:
        tester = np.multiply(t_nexts, increments)
        dims = np.where(tester == np.nanmin(tester))[0]
        assert len(dims) > 0
        t_nexts[dims] += np.multiply(steps[dims], increments[dims])
        if len(dims) > 1:
            combinations = get_all_combinations(dims)
            for axes_to_increment in combinations:
                n_pos = pos.copy()
                n_pos[axes_to_increment] += increments[axes_to_increment]
                if ((final_pos - n_pos) * increments >= 0).all():
                    yield n_pos.copy()
        
        pos[dims] += increments[dims]

        if (np.multiply((final_pos - pos), increments) >= 0).all():
            yield pos.copy()
        else:
            break


def get_points_from_bounds(xyzmin, xyzmax, start, end):
    assert len(xyzmin) == len(xyzmax) == len(start) == len(end)
    n_end = end - start
    n_start = start - start

    n_xyzmin = xyzmin - start
    n_xyzmax = xyzmax - start

    for dim in range(0, len(xyzmin)):
        if n_end[dim] == 0:
            continue

        if n_end[dim] > n_start[dim]:
            f_min = (max(n_xyzmin[dim], n_start[dim]) / n_end[dim])
            f_max = (min(n_xyzmax[dim], n_end[dim]) / n_end[dim])
        else:
            f_min = (min(n_xyzmax[dim], n_start[dim]) / n_end[dim])
            f_max = (max(n_xyzmin[dim], n_end[dim]) / n_end[dim])


        n_start = n_end * f_min
        n_end = n_end * f_max
    
    return (n_start + start, n_end + start)