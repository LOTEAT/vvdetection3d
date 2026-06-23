# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from shapely.geometry.polygon import Polygon

from vvdet3d.structures import Box3DMode, CameraInstance3DBoxes, points_cam2img
from vvdet3d.structures.ops import box_np_ops


# vvsim_categories = (
#     'car', 
#     'pedestrian', 
#     'truck', 
#     'speed_limit_sign',
#     'van', 
#     'trailer', 
#     'traffic_cone', 
# )

vvsim_categories = (
    'car', 
    'pedestrian', 
)

# VVSimNameMapping = {
#     'vehicle.car.suv': 'car',
#     'character.pedestrian.staff': 'pedestrian',
#     'vehicle.truck.pickup_truck': 'truck',
#     'vehicle.car.sedan': 'car',
#     'vehicle.car.sports_car': 'car',
#     'infrastructure.trafficSign.speed_limit_sign.25': 'speed_limit_sign',
#     'infrastructure.trafficSign.speed_limit_sign.35': 'speed_limit_sign',
#     'infrastructure.trafficSign.speed_limit_sign.60': 'speed_limit_sign',
#     'character.pedestrian.security.male': 'pedestrian',
#     'character.pedestrian.worker.female': 'pedestrian',
#     'character.pedestrian.staff.male': 'pedestrian',
#     'character.pedestrian.worker.female': 'pedestrian',
#     'character.pedestrian.worker.male': 'pedestrian',
#     "character.pedestrian.constructor.male": 'pedestrian',
#     "character.pedestrian.constructor.female": 'pedestrian',
#     'character.pedestrian.robot': 'pedestrian',
#     'vehicle.truck.construction_vehicle': 'construction_vehicle',
#     'vehicle.truck.semitrailer': 'trailer',
#     'vehicle.van': 'van',
#     'infrastructure.barrier.traffic_cone': 'traffic_cone',
# }


VVSimNameMapping = {
    'vehicle.car.suv': 'car',
    'character.pedestrian.staff': 'pedestrian',
    'vehicle.truck.pickup_truck': 'car',
    'vehicle.car.sedan': 'car',
    'vehicle.car.sports_car': 'car',
    'character.pedestrian.security.male': 'pedestrian',
    'character.pedestrian.worker.female': 'pedestrian',
    'character.pedestrian.staff.male': 'pedestrian',
    'character.pedestrian.worker.female': 'pedestrian',
    'character.pedestrian.worker.male': 'pedestrian',
    "character.pedestrian.constructor.male": 'pedestrian',
    "character.pedestrian.constructor.female": 'pedestrian',
    'character.pedestrian.robot': 'pedestrian',
    'vehicle.truck.semitrailer': 'car',
    'vehicle.van': 'car',
}



nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')
NuScenesNameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}
LyftNameMapping = {
    'bicycle': 'bicycle',
    'bus': 'bus',
    'car': 'car',
    'emergency_vehicle': 'emergency_vehicle',
    'motorcycle': 'motorcycle',
    'other_vehicle': 'other_vehicle',
    'pedestrian': 'pedestrian',
    'truck': 'truck',
    'animal': 'animal'
}

def get_vvsim_corners(bbox3d: np.ndarray) -> np.ndarray:
    x, y, z, l, w, h, yaw, _, _ = bbox3d

    x_corners = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2 ])
    y_corners = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2 ])
    z_corners = np.array([ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2 ])

    corners = np.vstack((x_corners, y_corners, z_corners))  # 3x8

    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    rotated_corners = R @ corners  # 3x8

    rotated_corners[0, :] += x
    rotated_corners[1, :] += y
    rotated_corners[2, :] += z
    return rotated_corners.T


def get_vvsim_2d_boxes(global2ego: np.ndarray, cams_info: dict, instances: list) -> List[dict]:
    """Get the 2d annotation records for vvsim dataset.

    Args:
        cam_info (dict): Camera information including the camera type and
            calibration information.
        instances (list): List of instances.

    Return:
        List[dict]: List of 2d annotation record.
    """
    bbox2d_instances = dict()
    for cam_type, cam_info in cams_info.items():
        bbox2d_instances[cam_type] = []
        for instance in instances:
            bbox3d = instance['bbox_3d']
            corners = get_vvsim_corners(bbox3d)
            cam2ego = cam_info['cam2ego']
            ego2cam = np.linalg.inv(cam2ego)
            num_pts = corners.shape[0]
            corners_hom = np.hstack([corners, np.ones((num_pts, 1))])
            corners_cam_hom = ego2cam @ global2ego @ corners_hom.T
            if not np.all(corners_cam_hom[2, :] > 0):
                continue
            # in_front = np.argwhere(corners_cam_hom[2, :] > 0).flatten()
            corners_3d = corners_cam_hom[:3, :]
            corners_2d = view_points(corners_3d, cam_info['cam2img'], True).T[:, :2].tolist()
            # Keep only corners that fall within the image.
            final_coords = post_process_coords(corners_2d, imsize=(cam_info['width'], cam_info['height']))

            if final_coords is None:
                continue
            else:
                min_x, min_y, max_x, max_y = final_coords

            # Generate dictionary record to be included in the .json file.
            repro_rec = generate_record(instance['label_name'], min_x, min_y, max_x, max_y,
                                    'vvsim')
            bbox2d_instances[cam_type].append(repro_rec)
    return bbox2d_instances

def post_process_coords(
    corner_coords: List[int], imsize: Tuple[int] = (1600, 900)
) -> Union[Tuple[float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (List[int]): Corner coordinates of reprojected
            bounding box.
        imsize (Tuple[int]): Size of the image canvas.
            Defaults to (1600, 900).

    Return:
        Tuple[float] or None: Intersection of the convex hull of the 2D box
        corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        if isinstance(img_intersection, Polygon):
            intersection_coords = np.array(
                [coord for coord in img_intersection.exterior.coords])
            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])
            return min_x, min_y, max_x, max_y
        else:
            warnings.warn('img_intersection is not an object of Polygon.')
            return None
    else:
        return None


def generate_record(cat_name: str, x1: float, y1: float, x2: float, y2: float,
                    dataset: str) -> Union[dict, None]:
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.

    Args:
        cat_name (str): Category name of the object.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        dataset (str): Name of dataset.

    Returns:
        dict or None: A sample 2d annotation record.

            - bbox_label (int): 2d box label id
            - bbox_label_3d (int): 3d box label id
            - bbox (List[float]): left x, top y, right x, bottom y of 2d box
            - bbox_3d_isvalid (bool): whether the box is valid
    """

    if dataset == 'vvsim':
        categories = vvsim_categories
    else:
        raise NotImplementedError('Unsupported dataset!')

    if cat_name not in categories:
        return None

    rec = dict()
    rec['bbox_label'] = categories.index(cat_name)
    rec['bbox'] = [x1, y1, x2, y2]
    return rec
