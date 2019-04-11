"""
HAKUNA MATATA
"""
from .convex_hull import ConvexHull
from .delaunay import Delaunay3D
from .kdtree import KDTree
from .voxelgrid import VoxelGrid
from .voxelgrid_old import VoxelGrid_Old
from .bounding_boxes import BoundingBoxes

ALL_STRUCTURES = {
    'convex_hull': ConvexHull,
    'delaunay3D': Delaunay3D,
    'kdtree': KDTree,
    'voxelgrid': VoxelGrid,
    'voxelgrid_old': VoxelGrid_Old,
    'bounding_boxes': BoundingBoxes
}
