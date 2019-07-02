import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np
import pythreejs

from base import Structure
from utils.matrices import roty, rotz, rotx

cur_id = 0

BBOX_COLORS = {
    'Car': '#ff0000',
    'Pedestrian': '#00ff00',
    'Van': '#ff8800',
    'Truck': '#ffff00',
    'Cyclist': '#0000ff',
    'Tram': '#00ffff',
    'Person (sitting)': '#00ff88',
    'Misc': '#88ff88'
}


def corners_to_centered_bbox(box3d_corner):
    # center point is in the center of xy plane and in bottom of z plane
    # (N, 8, 3) -> (N, 7)
    assert box3d_corner.ndim == 3
    batch_size = box3d_corner.shape[0]

    xyz = np.mean(box3d_corner[:, :4, :], axis=1)

    h = abs(np.mean(box3d_corner[:, 4:, 2] -
                    box3d_corner[:, :4, 2], axis=1, keepdims=True))

    w = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 1, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 2, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 5, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 6, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    l = (np.sqrt(np.sum((box3d_corner[:, 0, [0, 1]] - box3d_corner[:, 3, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 1, [0, 1]] - box3d_corner[:, 2, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 4, [0, 1]] - box3d_corner[:, 7, [0, 1]]) ** 2, axis=1, keepdims=True)) +
         np.sqrt(np.sum((box3d_corner[:, 5, [0, 1]] - box3d_corner[:, 6, [0, 1]]) ** 2, axis=1, keepdims=True))) / 4

    theta = (np.arctan2(box3d_corner[:, 2, 1] - box3d_corner[:, 1, 1],
                        box3d_corner[:, 2, 0] - box3d_corner[:, 1, 0]) +
             np.arctan2(box3d_corner[:, 3, 1] - box3d_corner[:, 0, 1],
                        box3d_corner[:, 3, 0] - box3d_corner[:, 0, 0]) +
             np.arctan2(box3d_corner[:, 2, 0] - box3d_corner[:, 3, 0],
                        box3d_corner[:, 3, 1] - box3d_corner[:, 2, 1]) +
             np.arctan2(box3d_corner[:, 1, 0] - box3d_corner[:, 0, 0],
                        box3d_corner[:, 0, 1] - box3d_corner[:, 1, 1]))[:, np.newaxis] / 4

    return np.concatenate([xyz, l, w, h, theta], axis=1).reshape(batch_size, 7)


class BoundingBoxes(Structure):
    def __init__(self, points, calib, bboxes, ignore_empty_bboxes=False, corners=None, **kwargs):
        ''' corners: list of bounding boxes with corners in the following order
            bboxes: info about the original bounding box Object3d (coordinates of this object aren't always right, they're based on original bounding box, so corners object should be trusted if available)
            6 -------- 7      z| x
           /|         /|       |/ 
          5 -------- 4 .  y -- -- -y
          | |        | |      /
          . 2 -------- 3    -x  
          |/         |/
          1 -------- 0
        '''
        Structure.__init__(self, points=points, **kwargs)
        self.calib = calib
        self.bboxes = bboxes
        if len(self.bboxes) > 0:
            if corners is not None:
                self.corners = corners
            else:
                self._compute_corners(bboxes)

            self.corners = np.array(self.corners)
            self.centered_bbox = corners_to_centered_bbox(
                self.corners)  # xyzlwhr
            self.bboxes = np.array(self.bboxes)

            if ignore_empty_bboxes is True:
                self.corners = self._filter_empty_bboxes(self.corners)
        else:
            self.corners = np.array([])
            self.centered_bbox = np.array([])

    def _filter_empty_bboxes(self, corners):
        # this code is here for reference, but in practice it's not efficient to calculate this each time we use the dataset
        # the right way to do it is to calculate it once and save the invalid bboxes in a file or remove them from the dataset directly
        # (unless the bounding boxes are dynamic)
        def point_inside_box(c, point):
            #  https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d
            ax = c[2] - c[1]
            ay = c[0] - c[1]
            az = c[5] - c[1]

            px = np.dot(ax, point)
            py = np.dot(ay, point)
            pz = np.dot(az, point)

            cx = px >= np.dot(ax, c[1]) and px <= np.dot(ax, c[2])
            cy = py >= np.dot(ay, c[1]) and py <= np.dot(ay, c[0])
            cz = pz >= np.dot(az, c[1]) and pz <= np.dot(az, c[5])

            return cx and cy and cz

        def box_contains_points(box, points):
            for point in points:
                if point_inside_box(box, point):
                    return True
            return False
        valid_corners = []
        for i, corner in enumerate(corners):
            if box_contains_points(corner, self._points) is True:
                valid_corners.append(i)

        return corners[valid_corners]

    def _compute_corners(self, bboxes):
        self.corners = []
        for obj in bboxes:
            # compute bboxes on REF COORD
            # compute rotational matrix around yaw axis
            R = roty(obj.ry)

            # 3d bounding box dimensions
            l = obj.l
            w = obj.w
            h = obj.h

            # 3d bounding box corners
            x_corners = [l / 2, l / 2, -l / 2, -
                         l / 2, l / 2, l / 2, -l / 2, -l / 2]
            y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
            z_corners = [w / 2, -w / 2, -w / 2,
                         w / 2, w / 2, -w / 2, -w / 2, w / 2]

            # rotate and translate 3d bounding box
            corners_3d = np.dot(R, np.vstack(
                [x_corners, y_corners, z_corners]))
            # print corners_3d.shape
            corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
            corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
            corners_3d[2, :] = corners_3d[2, :] + obj.t[2]

            # project from REF COORD to VELO COORD
            vel_corners = self.calib.project_ref_to_velo(
                np.transpose(corners_3d))
            self.corners.append(vel_corners)

    def compute(self):
        """ABC API"""
        global cur_id
        self.id = "BBOXES({})".format(cur_id)
        cur_id += 1

    def plot(self, scene):
        for cur in range(0, len(self.corners)):
            bbox_corners = self.corners[cur]
            bbox = self.bboxes[cur]
            lines = []
            for k in range(0, 4):
                i, j = k, (k + 1) % 4
                lines.append([(bbox_corners[i, 0], bbox_corners[i, 1], bbox_corners[i, 2]),
                              (bbox_corners[j, 0], bbox_corners[j, 1], bbox_corners[j, 2])])

                i, j = k + 4, (k + 1) % 4 + 4
                lines.append([(bbox_corners[i, 0], bbox_corners[i, 1], bbox_corners[i, 2]),
                              (bbox_corners[j, 0], bbox_corners[j, 1], bbox_corners[j, 2])])

                i, j = k, k + 4
                lines.append([(bbox_corners[i, 0], bbox_corners[i, 1], bbox_corners[i, 2]),
                              (bbox_corners[j, 0], bbox_corners[j, 1], bbox_corners[j, 2])])

            for line in lines:
                try:
                    color = BBOX_COLORS[bbox.type]
                except:
                    color = '#88ff88'

                line_geometry = pythreejs.Geometry(
                    vertices=line)
                drew_line = pythreejs.Line(
                    geometry=line_geometry,
                    material=pythreejs.LineBasicMaterial(color=color),
                    type='LinePieces')
                scene.add(drew_line)

    def generate_subcloud(self, box_xyzlwh):
        box_x, box_y, box_z, box_l, box_w, box_h = box_xyzlwh

        def point_inside_box(point):
            on_l = (point[0] <= box_x + box_l /
                    2) and (point[0] >= box_x - box_l / 2)
            on_w = (point[1] <= box_y + box_w /
                    2) and (point[1] >= box_y - box_w / 2)
            on_h = (point[2] <= box_z + box_h /
                    2) and (point[2] >= box_z - box_h / 2)
            return (on_l and on_w and on_h)

        idx_to_include = []
        for idx in range(0, len(self.corners)):
            include_corner_set = True
            corner_set = self.corners[idx]
            for i in range(0, len(corner_set)):
                if not point_inside_box(corner_set[i]):
                    include_corner_set = False
                    break
            if include_corner_set:
                idx_to_include.append(idx)

        new_corners = []
        for idx in idx_to_include:
            corner = self.corners[idx].copy()
            corner -= np.array([box_x, box_y, box_z])
            new_corners.append(corner)

        structure = BoundingBoxes(points=self._points, bounds=self.bounds, calib=self.calib, origin=self.origin,
                                  bboxes=self.bboxes[idx_to_include], corners=new_corners, ignore_empty_bboxes=False)
        return structure

    def __str__(self):
        st = ''

        st += '{} bounding box/es'.format(len(self.corners))
        return st

    def __len__(self):
        return len(self.corners)

    def get_shape(self, corner):
        w = np.linalg.norm(corner[0] - corner[1])
        h = np.linalg.norm(corner[5] - corner[1])
        l = np.linalg.norm(corner[2] - corner[1])
        return (l, w, h)

    def get_shapes(self):
        shapes = []
        for corner in self.corners:
            shapes.append(self.get_shape(corner))

        return shapes


if __name__ == '__main__':
    # test corners_to_centered_bbox
    # test bbox: bottom left corner on origin, width height and length of s, no angle
    s = 4
    corners = np.array([[[0, -s, 0], [0, 0, 0], [s, 0, 0], [s, -s, 0], 
                        [0, -s, s], [0, 0, s], [s, 0, s], [s, -s, s]]])
    print(corners_to_centered_bbox(corners))

    front_offset = 1
    corners = np.array([[[0, -s, 0], [0, 0, 0], [s, -front_offset, 0], [s, -s-front_offset, 0], 
                    [0, -s, s], [0, 0, s], [s, -front_offset, s], [s, -s-front_offset, s]]])
    print(corners_to_centered_bbox(corners))