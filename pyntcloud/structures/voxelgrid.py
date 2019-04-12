import numpy as np
import math
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


class VoxelGrid(Structure):

    def __init__(self, *, points, n_x=1, n_y=1, n_z=1, size_x=None, size_y=None, size_z=None):
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
        """
        super().__init__(points=points)

        if size_x is None or size_y is None or size_z is None:
            print('WARNING: when computing a voxelgrid for running a Neural net, voxel sizes should be homogeneous among different point clouds or the neural network wont learn spatial relationships. To ensure this, use (size_x, size_y, size_z) instead of (n_x, n_y, n_z)')

        self.x_y_z = [n_x, n_y, n_z]
        self.sizes = [size_x, size_y, size_z]

        self.xyzmin = self._points.min(0)
        self.xyzmax = self._points.max(0)
        
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

        assert voxel_x.max() < self.x_y_z[0], '{}, {}'.format(voxel_x.max(), self.x_y_z[0])
        assert voxel_y.max() < self.x_y_z[1], '{}, {}'.format(voxel_y.max(), self.x_y_z[1])
        assert voxel_z.max() < self.x_y_z[2], '{}, {}'.format(voxel_z.max(), self.x_y_z[2])

        return np.ravel_multi_index([voxel_x, voxel_y, voxel_z], self.x_y_z)

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

        self.voxel_n = self.locate_points(self._points)

        # TODO optimise this, now takes too long
        # compute center of each voxel
        # midsegments = [(self.segments[i][1:] + self.segments[i][:-1]) / 2 for i in range(3)]
        # self.voxel_centers = cartesian(midsegments).astype(np.float32)

    def query(self, points):
        """ABC API. Query structure.
        """
        return self.locate_points(points)

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
        vector = np.zeros(self.n_voxels)

        if mode == "binary":
            vector[np.unique(self.voxel_n)] = 1

        elif mode == "density":
            count = np.bincount(self.voxel_n)
            vector[:len(count)] = count
            vector /= len(self.voxel_n)

        # elif mode == "TDF":
        #     # truncation = np.linalg.norm(self.shape)
        #     kdt = cKDTree(self._points)
        #     vector, i = kdt.query(self.voxel_centers, n_jobs=-1)

        elif mode.endswith("_max"):
            if not is_numba_avaliable:
                raise ImportError("numba is required to compute {}".format(mode))
            axis = {"x_max": 0, "y_max": 1, "z_max": 2}
            vector = groupby_max(self._points, self.voxel_n, axis[mode], vector)

        elif mode.endswith("_mean"):
            if not is_numba_avaliable:
                raise ImportError("numba is required to compute {}".format(mode))
            axis = {"x_mean": 0, "y_mean": 1, "z_mean": 2}
            voxel_sum = groupby_sum(self._points, self.voxel_n, axis[mode], np.zeros(self.n_voxels))
            voxel_count = groupby_count(self._points, self.voxel_n, np.zeros(self.n_voxels))
            vector = np.nan_to_num(voxel_sum / voxel_count)

        else:
            raise NotImplementedError("{} is not a supported feature vector mode".format(mode))

        return vector.reshape(self.x_y_z)

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

        return [x for x in ravel_indices if x in np.unique(self.voxel_n)]

    def __str__(self):
        st = ''
        if self.segments:
            st += 'Num segments x: {}, y: {}, z: {}\n'.format(len(self.segments[0]), len(self.segments[1]), len(self.segments[2]))
        if self.shape:
            st += 'Shape of each voxel: {}\n'.format(self.shape)
        if self.n_voxels is not None:
            st += 'Total voxels: {}\n'.format(self.n_voxels)
        if self.voxel_n is not None:
            uniq, counts = np.unique(self.voxel_n, return_counts=True, axis=0)
            st += 'Max points in one single section: {}\n'.format(counts.max())
            st += 'Sections with at least one point: {}\n'.format(len(uniq))
        return st