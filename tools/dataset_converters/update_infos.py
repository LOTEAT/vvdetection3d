"""Convert the annotation pkl to the standard format.
"""

import copy
from os import path as osp
from pathlib import Path

import mmengine
import numpy as np
from nuscenes.nuscenes import NuScenes

from vvdet3d.datasets.convert_utils import (vvsim_categories,
                                            get_vvsim_2d_boxes)
from vvdet3d.datasets.utils import concat_rotation_translation
from vvdet3d.structures import points_cam2img


def transform_bbox(bbox_global, ego2global, lidar2ego):
    global2ego = np.linalg.inv(ego2global)
    ego2lidar = np.linalg.inv(lidar2ego)
    global2lidar = ego2lidar @ global2ego
    R = global2lidar[:3, :3] 
    x, y, z, yaw, vx, vy = bbox_global
    center_global_homogeneous = np.array([x, y, z, 1])
    center_lidar_homogeneous = global2lidar @ center_global_homogeneous
    new_x = center_lidar_homogeneous[0]
    new_y = center_lidar_homogeneous[1]
    new_z = center_lidar_homogeneous[2]
    direction_vec_global = np.array([np.cos(yaw), np.sin(yaw), 0])
    rotation_matrix = global2lidar[:3, :3]
    direction_vec_lidar = rotation_matrix @ direction_vec_global
    new_yaw = np.arctan2(direction_vec_lidar[1], direction_vec_lidar[0])

    v_global = np.array([vx, vy, 0.0])   # 全局坐标系速度
    v_lidar = R @ v_global
    new_vx, new_vy = v_lidar[0], v_lidar[1]
    return [new_x, new_y, new_z, new_yaw, new_vx, new_vy]




def get_empty_instance():
    """Empty annotation for single instance."""
    instance = dict(
        label_name=None,
        # (list[float], required): list of 4 numbers representing
        # the bounding box of the instance, in (x1, y1, x2, y2) order.
        bbox=None,
        # (int, required): an integer in the range
        # [0, num_categories-1] representing the category label.
        bbox_label=None,
        #  (list[float], optional): list of 7 (or 9) numbers representing
        #  the 3D bounding box of the instance,
        #  in [x, y, z, w, h, l, yaw]
        #  (or [x, y, z, w, h, l, yaw, vx, vy]) order.
        bbox_3d=None,
        # (bool, optional): Whether to use the
        # 3D bounding box during training.
        bbox_3d_isvalid=None,
        # (int, optional): 3D category label
        # (typically the same as label).
        bbox_label_3d=None,
        # (float, optional): Projected center depth of the
        # 3D bounding box compared to the image plane.
        depth=None,
        #  (list[float], optional): Projected
        #  2D center of the 3D bounding box.
        center_2d=None,
        # (int, optional): Attribute labels
        # (fine-grained labels such as stopping, moving, ignore, crowd).
        attr_label=None,
        # (int, optional): The number of LiDAR
        # points in the 3D bounding box.
        num_lidar_pts=None,
        # (int, optional): The number of Radar
        # points in the 3D bounding box.
        num_radar_pts=None,
        # (int, optional): Difficulty level of
        # detecting the 3D bounding box.
        difficulty=None,
        unaligned_bbox_3d=None)
    return instance


def get_empty_lidar_points():
    lidar_points = dict(
        # (int, optional) : Number of features for each point.
        num_pts_feats=None,
        # (str, optional): Path of LiDAR data file.
        lidar_path=None,
        # (list[list[float]], optional): Transformation matrix
        # from lidar to ego-vehicle
        # with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        lidar2ego=None,
    )
    return lidar_points


def get_empty_radar_points():
    radar_points = dict(
        # (int, optional) : Number of features for each point.
        num_pts_feats=None,
        # (str, optional): Path of RADAR data file.
        radar_path=None,
        # Transformation matrix from lidar to
        # ego-vehicle with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        radar2ego=None,
    )
    return radar_points


def get_empty_img_info():
    img_info = dict(
        # (str, required): the path to the image file.
        img_path=None,
        # (int) The height of the image.
        height=None,
        # (int) The width of the image.
        width=None,
        # (str, optional): Path of the depth map file
        depth_map=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to image with
        # shape [3, 3], [3, 4] or [4, 4].
        cam2img=None,
        # (list[list[float]]): Transformation matrix from lidar
        # or depth to image with shape [4, 4].
        lidar2img=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to ego-vehicle
        # with shape [4, 4].
        cam2ego=None)
    return img_info


def get_single_image_sweep(camera_types):
    single_image_sweep = dict(
        # (float, optional) : Timestamp of the current frame.
        timestamp=None,
        # (list[list[float]], optional) : Transformation matrix
        # from ego-vehicle to the global
        ego2global=None)
    # (dict): Information of images captured by multiple cameras
    images = dict()
    for cam_type in camera_types:
        images[cam_type] = get_empty_img_info()
    single_image_sweep['images'] = images
    return single_image_sweep


def get_single_lidar_sweep():
    single_lidar_sweep = dict(
        # (float, optional) : Timestamp of the current frame.
        timestamp=None,
        # (list[list[float]], optional) : Transformation matrix
        # from ego-vehicle to the global
        ego2global=None,
        # (dict): Information of images captured by multiple cameras
        lidar_points=get_empty_lidar_points())
    return single_lidar_sweep


def get_empty_standard_data_info(
        camera_types=['CAM0', 'CAM1', 'CAM2', 'CAM3', 'CAM4']):

    data_info = dict(
        **get_single_image_sweep(camera_types),
        # (dict, optional): dict contains information
        # of LiDAR point cloud frame.
        lidar_points=get_empty_lidar_points(),
        # (dict, optional) Each dict contains
        # information of Radar point cloud frame.
        radar_points=get_empty_radar_points())
    return data_info

def get_empty_instance_data_info():
    instance_info = dict(
        instances_global = [],
        instances_local = dict(),
        ignore_instances = [],
        cam_instances = {})
    return instance_info


def clear_instance_unused_keys(instance):
    keys = list(instance.keys())
    for k in keys:
        if instance[k] is None:
            del instance[k]
    return instance


def clear_data_info_unused_keys(data_info):
    keys = list(data_info.keys())
    empty_flag = True
    for key in keys:
        # we allow no annotations in datainfo
        if key in ['instances', 'cam_sync_instances', 'cam_instances']:
            empty_flag = False
            continue
        if isinstance(data_info[key], list):
            if len(data_info[key]) == 0:
                del data_info[key]
            else:
                empty_flag = False
        elif data_info[key] is None:
            del data_info[key]
        elif isinstance(data_info[key], dict):
            _, sub_empty_flag = clear_data_info_unused_keys(data_info[key])
            if sub_empty_flag is False:
                empty_flag = False
            else:
                # sub field is empty
                del data_info[key]
        else:
            empty_flag = False

    return data_info, empty_flag


def update_vvsim_infos(pkl_path, out_dir):
    cav_camera_types = [
        'CAM_FRONT',
        'CAM_RIGHT',
        'CAM_LEFT',
        'CAM_REAR',
    ]
    drone_camera_types = ['CAM_BOTTOM']
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
    print(f'Reading from input file: {pkl_path}.')
    scene_infos = mmengine.load(pkl_path)
    METAINFO = {
        'classes': vvsim_categories,
    }
    print('Start updating:')
    converted_scene_infos = []
    for scene_info in mmengine.track_iter_progress(scene_infos):
        converted_scene_info = {
            'data_list': [],
            'metainfo': scene_info['metainfo']
        }
        converted_data_list = []
        # TODO: remove this line
        for ori_agent_dict in mmengine.track_iter_progress(scene_info['data_list']):
            new_ori_agent_dict = copy.deepcopy(ori_agent_dict)
            new_ori_agent_dict['data_dict'].update(get_empty_instance_data_info())
            if 'gt_boxes_global' not in ori_agent_dict['data_dict']:
                raise ValueError('gt_boxes_global should be provided in the original pkl file.')
            else:
                ignore_class_name = set()
                num_instances = ori_agent_dict['data_dict']['gt_boxes_global'].shape[0]
                for i in range(num_instances):
                    empty_instance = get_empty_instance()
                    empty_instance['bbox_3d'] = ori_agent_dict['data_dict']['gt_boxes_global'][
                        i, :].tolist()
                    empty_instance['bbox_3d'] += ori_agent_dict['data_dict']['gt_velocity_global'][i, :].tolist()
                    if ori_agent_dict['data_dict']['gt_names'][i] in METAINFO['classes']:
                        label_name = ori_agent_dict['data_dict']['gt_names'][i]
                        empty_instance['label_name'] = label_name
                        empty_instance['bbox_label'] = METAINFO['classes'].index(label_name)
                    else:
                        ignore_class_name.add(ori_agent_dict['data_dict']['gt_names'][i])
                        empty_instance['bbox_label'] = -1
                    empty_instance['bbox_label_3d'] = copy.deepcopy(
                        empty_instance['bbox_label'])
                    empty_instance = clear_instance_unused_keys(empty_instance)
                    new_ori_agent_dict['data_dict']['instances_global'].append(empty_instance)
                
                
                    
            for agent in ori_agent_dict['metainfo']['agents']:
                ori_agent_info = ori_agent_dict['data_dict'][agent]
                if 'cav' in agent.lower():
                    camera_types = cav_camera_types
                elif 'drone' in agent.lower():
                    camera_types = drone_camera_types
                else:
                    raise NotImplementedError(
                        f'Unknown agent type: {agent}. '
                        f'Only support cav and drone now.')
                
                temp_data_info = get_empty_standard_data_info(
                    camera_types=camera_types)
                temp_data_info['position'] = ori_agent_info['position']
                temp_data_info['ego2global'] = concat_rotation_translation(
                    ori_agent_info['ego2global_rotation'],
                    ori_agent_info['ego2global_translation'])
                if 'cav' in agent.lower():
                    temp_data_info['lidar_points']['num_pts_feats'] = ori_agent_info.get(
                        'num_features', 5)
                    temp_data_info['lidar_points']['lidar_path'] = ori_agent_info['lidar_path']
                    temp_data_info['lidar_points'][
                        'lidar2ego'] = concat_rotation_translation(
                        ori_agent_info['lidar2ego_rotation'],
                        ori_agent_info['lidar2ego_translation'])
                for cam in ori_agent_info['cams']:
                    empty_img_info = get_empty_img_info()
                    empty_img_info['height'] = ori_agent_info['cams'][cam]['height']
                    empty_img_info['width'] = ori_agent_info['cams'][cam]['width']
                    empty_img_info['img_path'] = ori_agent_info['cams'][cam]['image_path']
                    
                    empty_img_info['cam2img'] = np.array(ori_agent_info['cams'][cam][
                        'cam_intrinsic'], dtype=np.float32)
                    empty_img_info['cam2ego'] = concat_rotation_translation(
                        ori_agent_info['cams'][cam]['sensor2ego_rotation'],
                        ori_agent_info['cams'][cam]['sensor2ego_translation'])
                    if 'cav' in agent.lower():
                        lidar2sensor = np.eye(4)
                        rot = ori_agent_info['cams'][cam]['sensor2top_rotation']
                        trans = ori_agent_info['cams'][cam]['sensor2top_translation']
                        lidar2sensor[:3, :3] = rot.T
                        lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
                        empty_img_info['lidar2cam'] = lidar2sensor.astype(
                            np.float32).tolist()
                    temp_data_info['images'][cam] = empty_img_info
                temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
                new_ori_agent_dict['data_dict'][agent] = temp_data_info
                global2ego = np.linalg.inv(temp_data_info['ego2global'])
                new_ori_agent_dict['data_dict']['cam_instances'][agent] = get_vvsim_2d_boxes(
                    global2ego, temp_data_info['images'], new_ori_agent_dict['data_dict']['instances_global'])
                new_ori_agent_dict['data_dict']['instances_local'].setdefault(agent, [])
                for instance_global in new_ori_agent_dict['data_dict']['instances_global']:
                    instance_local = copy.deepcopy(instance_global)
                    bbox_3d_global = instance_local['bbox_3d']
                    if 'cav' in agent.lower():
                        bbox_3d_local = transform_bbox(
                            [bbox_3d_global[0], bbox_3d_global[1], bbox_3d_global[2], bbox_3d_global[6], bbox_3d_global[7], bbox_3d_global[8]],
                            new_ori_agent_dict['data_dict'][agent]['ego2global'],
                            new_ori_agent_dict['data_dict'][agent]['lidar_points']['lidar2ego'])
                    elif 'drone' in agent.lower():
                        lidar2ego = np.eye(4)
                        bbox_3d_local = transform_bbox(
                            [bbox_3d_global[0], bbox_3d_global[1], bbox_3d_global[2], bbox_3d_global[6], bbox_3d_global[7], bbox_3d_global[8]],
                            new_ori_agent_dict['data_dict'][agent]['ego2global'],
                            lidar2ego)
                    
                    instance_local['bbox_3d'] = [
                        bbox_3d_local[0], bbox_3d_local[1], bbox_3d_local[2],
                        bbox_3d_global[3], bbox_3d_global[4], bbox_3d_global[5],
                        bbox_3d_local[3], bbox_3d_local[4], bbox_3d_local[5]]
                    new_ori_agent_dict['data_dict']['instances_local'][agent].append(instance_local)
                
            converted_data_list.append(new_ori_agent_dict)
        converted_scene_info['data_list'] = converted_data_list
        converted_scene_infos.append(converted_scene_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')



    metainfo = dict()
    metainfo['categories'] = {class_name: idx for idx, class_name in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'vvsim'
    metainfo['scene_length'] = sum([scene_info['metainfo']['scene_length'] for scene_info in converted_scene_infos])
    metainfo['frame_gap'] = 40.0 # miliseconds
    data_infos = dict(metainfo=metainfo, scene_list=converted_scene_infos)

    mmengine.dump(data_infos, out_path, 'pkl')



def update_pkl_infos(dataset, out_dir, pkl_path):
    if dataset.lower() == 'vvsim':
        update_vvsim_infos(pkl_path=pkl_path, out_dir=out_dir)
    else:
        raise NotImplementedError(f'Do not support convert {dataset}.')

