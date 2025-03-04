import time
import numpy as np
import math
from ..utils.voxelgrid import supercover_line, get_points_from_bounds
try:
    import matplotlib.pyplot as plt
    is_matplotlib_avaliable = True
except ImportError:
    is_matplotlib_avaliable = False

from scipy.spatial import cKDTree

from .base import Structure
from ..plot import plot_voxelgrid
from ..utils.array import cartesian

try:
    from ..utils.numba import groupby_max, groupby_count, groupby_sum
    is_numba_avaliable = True
except ImportError:
    is_numba_avaliable = False

PRECISION = 1e-6

# TODO big refactoring

class VoxelGrid(Structure):
    def __init__(self, *, points, n_x=1, n_y=1, n_z=1, size_x=None, size_y=None, size_z=None, range=None, **kwargs):
        """Grid of voxels with support for different build methods.
            
        Parameters
        ----------
        points: (N, 3) numpy.array
        n_x, n_y, n_z :  int, optional
            Default: 1
            The number of segments in which each axis will be divided.
            Ignored if corresponding size_x, size_y or size_z is not None.
        size_x, size_y, size_z : float, optional
            Default: None
            The desired voxel size along each axis.
            If not None, the corresponding n_x, n_y or n_z will be ignored.
        range: list of floats
            f.ex: [0, -40, -3, 70, 40, 1]
            default: None
            format: [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        super().__init__(points=points, **kwargs)

        if size_x is None or size_y is None or size_z is None:
            print('WARNING: when computing a voxelgrid for running a Neural net, voxel sizes should be homogeneous among different point clouds or the neural network wont learn spatial relationships. To ensure this, use (size_x, size_y, size_z) instead of (n_x, n_y, n_z)')

        self.x_y_z = [n_x, n_y, n_z]
        self.sizes = np.array([size_x, size_y, size_z])

        if range is None:
            self.xyzmin = self.bounds[0]
            self.xyzmax = self.bounds[1]
        else: 
            self.xyzmin = range[:3]
            self.xyzmax = range[3:]

        for n, size in enumerate(self.sizes):
            if size is None:
                continue

            # ensure that 'sizes' are respected by making the box bigger if necessary
            margin = size - ((self.xyzmax[n] - self.xyzmin[n]) % size)
            self.xyzmin[n] -= margin / 2
            self.xyzmax[n] += margin / 2
            self.x_y_z[n] = int(round((self.xyzmax[n] - self.xyzmin[n]) / size))

    def locate_points(self, points):
        # find where each point lies in corresponding segmented axis
        if self.segments is None:
            raise Exception('call compute first')

        # px = points[:, 0]
        # py = points[:, 1]
        # pz = points[:, 2]

        # s_px = np.argsort(px)
        # s_py = np.argsort(py)
        # s_pz = np.argsort(pz)

        # res = np.zeros_like(points, dtype=int)

        # pointer = 0
        # for ind_x in s_px:
        #     px_v = px[ind_x]
        #     while pointer < len(self.segments[0]) and self.segments[0][pointer] <= px_v:
        #         pointer += 1
        #     pointer -= 1
        #     res[ind_x,0] = pointer
        #     assert pointer < self.x_y_z[0], 'index: {}, max: {}'.format(pointer, self.x_y_z[0])

        # pointer = 0 
        # for ind_y in s_py:
        #     py_v = py[ind_y]
        #     while pointer < len(self.segments[1]) and self.segments[1][pointer] <= py_v:
        #         pointer += 1
        #     pointer -= 1
        #     res[ind_y,1] = pointer

        #     assert pointer < self.x_y_z[1], 'index: {}, max: {}'.format(pointer, self.x_y_z[1])

        # pointer = 0 
        # for ind_z in s_pz:
        #     pz_v = pz[ind_z]
        #     while pointer < len(self.segments[2]) and self.segments[2][pointer] <= pz_v:
        #         pointer += 1
        #     pointer -= 1
        #     res[ind_z,2] = pointer

        #     assert pointer < self.x_y_z[2], 'index: {}, max: {}'.format(pointer, self.x_y_z[2])

        voxel_x = np.searchsorted(self.segments[0], points[:, 0], side='right') - 1
        voxel_y = np.searchsorted(self.segments[1], points[:, 1], side='right') - 1
        voxel_z = np.searchsorted(self.segments[2], points[:, 2], side='right') - 1

        self.is_inside_bounds = (voxel_x >= 0) & (voxel_x < self.x_y_z[0]-1) & (voxel_y >= 0) & (voxel_y < self.x_y_z[1]-1) & (voxel_z >= 0) & (voxel_z < self.x_y_z[2]-1)
        
        self.voxel_x = voxel_x[self.is_inside_bounds].astype(np.int32)  
        self.voxel_y = voxel_y[self.is_inside_bounds].astype(np.int32)  
        self.voxel_z = voxel_z[self.is_inside_bounds].astype(np.int32)

        self.points_inside_bounds = points[self.is_inside_bounds]

        assert self.voxel_x.max() < self.x_y_z[0]-1, '{}, {}'.format(self.voxel_x.max(), self.x_y_z[0])
        assert self.voxel_y.max() < self.x_y_z[1]-1, '{}, {}'.format(self.voxel_y.max(), self.x_y_z[1])
        assert self.voxel_z.max() < self.x_y_z[2]-1, '{}, {}'.format(self.voxel_z.max(), self.x_y_z[2])

    def compute(self):
        """ABC API."""
        segments = []
        shape = []
        for i in range(3):
            # note the +1 in num
            s, step = np.linspace(self.xyzmin[i], self.xyzmax[i], num=(self.x_y_z[i]), retstep=True, endpoint=False)
            segments.append(s)
            shape.append(step)

            if self.sizes[i] is not None:
                assert abs(self.sizes[i] - step) <= 0.00001, 'given voxel sizes are not being respected: {} should be {}'.format(step, self.sizes[i])

            assert len(s) == self.x_y_z[i]
        self.segments = segments
        self.shape = shape

        self.n_voxels = len(self.segments[0]) * len(self.segments[1]) * len(self.segments[2])

        self.id = "V({},{})".format(self.x_y_z, self.sizes)

        self.locate_points(self._points)

        # TODO optimise this, now takes too long
        # compute center of each voxel
        # midsegments = [(self.segments[i][1:] + self.segments[i][:-1]) / 2 for i in range(3)]
        # self.voxel_centers = cartesian(midsegments).astype(np.float32)

    def query(self, points):
        """ABC API. Query structure.
        """
        return self.locate_points(points)

    # TODO implement sparse version of other modes
    # TODO store already computed features
    def get_sparse_features(self, mode='binary'):
        if mode == 'binary':
            indices = np.stack([self.voxel_x, self.voxel_y, self.voxel_z], axis=1)
            features = np.ones(indices.shape[0], dtype=np.uint8)
            return (indices, features)
        else:
            raise Exception('Invalid model: {}'.format(mode))

    def get_feature_vector(self, mode="binary"):
        """Return a vector of size self.n_voxels. See mode options below.

        Parameters
        ----------
        mode: str in available modes. See Notes
            Default "binary"

        Returns
        -------
        feature_vector: [n_x, n_y, n_z] ndarray
            See Notes.

        Notes
        -----
        Available modes are:

        binary
            0 for empty voxels, 1 for occupied.

        binary_with_nopoints
            0 for empty voxels, 1 for occupied, -1 for voxels through which rays pass (nopoints)

        density
            number of points inside voxel / total number of points.

        TDF
            Truncated Distance Function. Value between 0 and 1 indicating the distance
            between the voxel's center and the closest point. 1 on the surface,
            0 on voxels further than 2 * voxel side.

        x_max, y_max, z_max
            Maximum coordinate value of points inside each voxel.

        x_mean, y_mean, z_mean
            Mean coordinate value of points inside each voxel.
        """
        voxel_n = np.ravel_multi_index([self.voxel_x, self.voxel_y, self.voxel_z], self.x_y_z)
        if mode == "binary":
            vector = np.zeros(self.n_voxels)
            vector[np.unique(voxel_n)] = 1
            vector = vector.reshape(self.x_y_z)
            return vector

        elif mode == "binary_with_nopoints":
            vector = np.zeros(self.n_voxels)
            vector[np.unique(voxel_n)] = 1
            vector = vector.reshape(self.x_y_z)
            tot_bounds = abs(self.bounds[0]) + abs(self.bounds[1])
            # TODO can be parallelised
            non_points = []
            for point in self.points_inside_bounds:
                start, end = get_points_from_bounds(self.bounds[0], self.bounds[1], self.origin, point)
                start_projected_voxelgrid = (start - self.bounds[0])
                end_projected_voxelgrid = (end - self.bounds[0])

                assert np.all(start_projected_voxelgrid + PRECISION >= 0), 'Start / end point for nopoints calculation out of bounds: {} / {}'.format(start_projected_voxelgrid + PRECISION, tot_bounds)
                assert np.all(end_projected_voxelgrid + PRECISION >= 0), 'Start / end point for nopoints calculation out of bounds: {} / {}'.format(end_projected_voxelgrid + PRECISION, tot_bounds)
                assert np.all(start_projected_voxelgrid - PRECISION <= tot_bounds), 'Start / end point for nopoints calculation out of bounds: {} / {}'.format(start_projected_voxelgrid, tot_bounds)
                assert np.all(end_projected_voxelgrid - PRECISION <= tot_bounds), 'Start / end point for nopoints calculation out of bounds: {} / {}'.format(end_projected_voxelgrid, tot_bounds)

                start_projected_voxelgrid = np.clip(start_projected_voxelgrid, 0, tot_bounds - PRECISION)
                end_projected_voxelgrid = np.clip(end_projected_voxelgrid, 0, tot_bounds - PRECISION)

                new_non_points = list(supercover_line(start_projected_voxelgrid, end_projected_voxelgrid, self.sizes))
                non_points.extend(new_non_points)
                # if not np.all(np.array(new_non_points) >= 0) or not np.all(np.array(new_non_points).max(axis=0) < vector.shape):
                #     print('Non-point detected with indices under 0 or over size')
                #     print('start = {}'.format(start_projected_voxelgrid))
                #     print('end = {}'.format(end_projected_voxelgrid))
                #     print('Max Size: {}'.format(vector.shape))
                #     print('Wrong points:')
                #     print(np.array(new_non_points))
                #     raise Exception()

            # convert only cells that are 0 to -1, NOT 1 to -1
            non_points = np.unique(np.array(non_points), axis=0).astype(int)

            temp = vector[non_points[:, 0], non_points[:, 1], non_points[:, 2]]
            temp[temp == 0] = -1
            vector[non_points[:, 0], non_points[:, 1], non_points[:, 2]] = temp
            return vector
        elif mode == "density":
            vector = np.zeros(self.n_voxels)
            count = np.bincount(voxel_n)
            vector[:len(count)] = count
            vector /= len(voxel_n)
            vector = vector.reshape(self.x_y_z)
            return vector
        # elif mode == "TDF":
        #     vector = np.zeros(self.n_voxels)
        #     # truncation = np.linalg.norm(self.shape)
        #     kdt = cKDTree(self.points_inside_bounds)
        #     vector, i = kdt.query(self.voxel_centers, n_jobs=-1)
        #     vector = vector.reshape(self.x_y_z)
        #     return vector
        elif mode.endswith("_max"):
            vector = np.zeros(self.n_voxels)
            if not is_numba_avaliable:
                raise ImportError("numba is required to compute {}".format(mode))
            axis = {"x_max": 0, "y_max": 1, "z_max": 2}
            vector = groupby_max(self.points_inside_bounds, voxel_n, axis[mode], vector)
            vector = vector.reshape(self.x_y_z)
            return vector
        elif mode.endswith("_mean"):
            vector = np.zeros(self.n_voxels)
            if not is_numba_avaliable:
                raise ImportError("numba is required to compute {}".format(mode))
            axis = {"x_mean": 0, "y_mean": 1, "z_mean": 2}
            voxel_sum = groupby_sum(self.points_inside_bounds, voxel_n, axis[mode], np.zeros(self.n_voxels))
            voxel_count = groupby_count(self.points_inside_bounds, voxel_n, np.zeros(self.n_voxels))
            vector = np.nan_to_num(voxel_sum / voxel_count)
            vector = vector.reshape(self.x_y_z)
            return vector

        else:
            raise NotImplementedError("{} is not a supported feature vector mode".format(mode))


    def get_voxel_neighbors(self, voxel):
        """Get valid, non-empty 26 neighbors of voxel.

        Parameters
        ----------
        voxel: int in self.set_voxel_n

        Returns
        -------
        neighbors: list of int
            Indices of the valid, non-empty 26 neighborhood around voxel.
        """

        x, y, z = np.unravel_index(voxel, self.x_y_z)

        valid_x = []
        valid_y = []
        valid_z = []
        if x - 1 >= 0:
            valid_x.append(x - 1)
        if y - 1 >= 0:
            valid_y.append(y - 1)
        if z - 1 >= 0:
            valid_z.append(z - 1)

        valid_x.append(x)
        valid_y.append(y)
        valid_z.append(z)

        if x + 1 < self.x_y_z[0]:
            valid_x.append(x + 1)
        if y + 1 < self.x_y_z[1]:
            valid_y.append(y + 1)
        if z + 1 < self.x_y_z[2]:
            valid_z.append(z + 1)

        valid_neighbor_indices = cartesian((valid_x, valid_y, valid_z))

        ravel_indices = np.ravel_multi_index((valid_neighbor_indices[:, 0],
                                              valid_neighbor_indices[:, 1],
                                              valid_neighbor_indices[:, 2]), self.x_y_z)
        voxel_n = np.ravel_multi_index([self.voxel_x, self.voxel_y, self.voxel_z], self.x_y_z)
        return [x for x in ravel_indices if x in np.unique(voxel_n)]

    def __str__(self):
        st = ''
        if self.segments:
            st += 'Num segments x: {}, y: {}, z: {}\n'.format(len(self.segments[0]), len(self.segments[1]), len(self.segments[2]))
        if self.shape:
            st += 'Shape of each voxel: {}\n'.format(self.shape)
        if self.n_voxels is not None:
            st += 'Total voxels: {}\n'.format(self.n_voxels)
        # if self.voxel_n is not None:
        #     uniq, counts = np.unique(self.voxel_n, return_counts=True, axis=0)
        #     st += 'Max points in one single section: {}\n'.format(counts.max())
        #     st += 'Sections with at least one point: {}\n'.format(len(uniq))
        st+= 'Boundaries: min: {}, max: {}'.format(self.xyzmin, self.xyzmax)
        return st