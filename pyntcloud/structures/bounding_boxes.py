import pandas as pd
import numpy as np
import pythreejs

from .base import Structure
from ..utils.matrices import roty, rotz, rotx

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


class BoundingBoxes(Structure):
    def __init__(self, points, calib, bboxes, corners=None):
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
        Structure.__init__(self, points=points)
        self.calib = calib
        self.bboxes = bboxes
        if corners is not None:
            self.corners = corners
        else:
            self._compute_corners(bboxes)
        self.corners = np.array(self.corners)
        self.bboxes = np.array(self.bboxes)

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
            on_l = (point[0] <= box_x + box_l/2) and (point[0] >= box_x - box_l/2)
            on_w = (point[1] <= box_y + box_w/2) and (point[1] >= box_y - box_w/2)
            on_h = (point[2] <= box_z + box_h/2) and (point[2] >= box_z - box_h/2)
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

        structure = BoundingBoxes(points=self._points, calib=self.calib, bboxes=self.bboxes[idx_to_include], corners=new_corners)
        return structure

    def __str__(self):
        st = ''

        st += '{} bounding box/es'.format(len(self.corners))
        return st

    def __len__(self):
        return len(self.corners)