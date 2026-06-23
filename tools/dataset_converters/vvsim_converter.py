import os
from os import path as osp
import mmengine
from mmengine import track_iter_progress, track_parallel_progress
import numpy as np
from scipy.spatial.transform import Rotation as R
from vvdet3d.datasets.convert_utils import VVSimNameMapping

def create_vvsim_infos(root_path,
                        info_prefix,
                        data_split='train',
                        max_sweeps=10,
                        workers=4):
    """Create info file of vvsim dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        data_split (str, optional): The split of dataset, e.g., 'train', 'val', 'test', 'trainval'.
            Default: 'train'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
        workers (int): Number of threads to be used.
    """
    available_data_split = ['train', 'val', 'test', 'trainval', 'demo']

    # def get_scenes(data_split):
    #     folder_path = osp.join(root_path, data_split)
    #     assert mmengine.isdir(
    #         folder_path
    #     ), f'FileNotFoundError: No such directory: \'{data_split}\''
    #     return sorted([
    #         osp.join(folder_path, scene) for scene in os.listdir(folder_path)
    #     ])

    def get_scenes(data_split):
        folder_path = osp.join(root_path, data_split)
        assert mmengine.isdir(
            folder_path
        ), f'FileNotFoundError: No such directory: \'{data_split}\''
        folders = []
        for scene in os.listdir(folder_path):
            folders.append(osp.join(folder_path, scene))
        return sorted(folders)
        
    assert data_split in available_data_split
    if data_split == 'trainval':
        scenes = get_scenes('train') + get_scenes('val')
    else:
        scenes = get_scenes(data_split)
    print('{} scenes: {}'.format(data_split, len(scenes)))
    vvsim_data_infos = []
    if workers == 0:
        for scene in track_iter_progress(scenes):
            vvsim_data_infos.append(_fill_infos(scene))
    else:
        vvsim_data_infos = track_parallel_progress(_fill_infos, scenes, nproc=workers)
    info_path = osp.join(root_path,
                        '{}_infos_{}.pkl'.format(info_prefix, data_split))
    mmengine.dump(vvsim_data_infos, info_path)


def _fill_infos(scene):
    """Generate the train/val/test/trainval infos from the raw data.

    Args:
        scenes (list[str]): Basic information of scenes.

    Returns:
        OrderedDict: Information of train/val/test/trainval set,
        that will be saved to the info file.
    """
    cav_camera_types = ['CAM_FRONT', 'CAM_REAR', 'CAM_LEFT', 'CAM_RIGHT']
    drone_camera_types = ['CAM_BOTTOM']
    timestamps = sorted(os.listdir(scene))
    scene_dirs = [d for d in timestamps if osp.isdir(osp.join(scene, d))]

    metainfo = {
        'scene_name': osp.split(scene)[-1],
        'scene_length': len(scene_dirs),  # 只统计子文件夹数量
    }

    scene_info = {
        'data_list' : [],
        'metainfo': metainfo
    }
    # TODO: remove this line
    for timestamp in timestamps:
        if 'metadata' in timestamp:
            continue
        timestamp_info = {
            'metainfo': {
                'timestamp': timestamp,
                'agents': [],
            },
            'data_dict': {
                'gt_boxes_global': np.empty((0, 7), dtype=np.float32),
                'gt_names': np.empty((0,), dtype='<U20'),
                'gt_velocity_global': np.empty((0, 2), dtype=np.float32),
            }
        }
        timestamp_dir = osp.join(scene, timestamp)
        anno_path = osp.join(timestamp_dir, f'{timestamp}_annos.json')
        annos = mmengine.load(anno_path)
        for agent_name in os.listdir(timestamp_dir):
            agent_path = osp.join(timestamp_dir, agent_name)
            if not mmengine.isdir(agent_path):
                continue
            timestamp_info['metainfo']['agents'].append(agent_name)

            agent_anno = None
            
            if 'cav' in agent_name.lower():
                agent_type = 'cav'
                for anno in annos['agentInfo']:
                    if anno['agentName'] == agent_name:
                        agent_anno = anno
                        break
                lidar_path = osp.join(agent_path, f'LIDAR_TOP.pcd.bin')
                position = agent_anno['localization']['pose']['position']
                orientation = agent_anno['localization']['pose']['orientation']
                lidar2ego_translation = np.array(agent_anno['lidar']['LIDAR_TOP']['translation']).astype(np.float32)
                lidar2ego_rotation = np.array(agent_anno['lidar']['LIDAR_TOP']['rotation']).astype(np.float32)
                
                # FLU -> FRD
                lidar2ego_rotation = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ]) @ lidar2ego_rotation
                                
                ego2global_translation = np.array([position['x'], position['y'], position['z']]).astype(np.float32)
                ego2global_rotation = R.from_quat([orientation['qx'], orientation['qy'], orientation['qz'], orientation['qw']]).as_matrix().astype(np.float32)
               # FRD -> RFU
                ego2global_rotation = ego2global_rotation @ np.array(
                    [[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, -1]]
                ).astype(np.float32)
                agent_pos = np.array(
                    [position['x'], position['y'], position['z']]).astype(np.float32)
                agent_info = {
                    'lidar_path': lidar_path,
                    'cams': dict(),
                    'sweeps': [],
                    'num_features': 3,
                    'lidar2ego_translation': lidar2ego_translation,
                    'lidar2ego_rotation': lidar2ego_rotation,
                    'ego2global_translation': ego2global_translation,
                    'ego2global_rotation': ego2global_rotation,
                    'agent_type': agent_type,
                    'agent_name': agent_name,
                    'position': agent_pos,
                } 
                for cam in cav_camera_types: 
                    cam_path = osp.join(agent_path, f'{cam}_RGB.jpg')
                    cam_info = dict(image_path=cam_path)
                    cam_info['height'] = agent_anno['camera'][cam]['height']
                    cam_info['width'] = agent_anno['camera'][cam]['width']
                    cam_info.update(obtain_sensor2top(
                        agent_anno['camera'][cam]['translation'],
                        agent_anno['camera'][cam]['rotation'],
                        lidar2ego_translation,
                        lidar2ego_rotation))
                    cam_info['sensor2ego_translation'] = agent_anno['camera'][cam]['translation']
                    cam_info['sensor2ego_rotation'] = agent_anno['camera'][cam]['rotation']
                    intrinsic = agent_anno['camera'][cam]['intrinsic']
                    cam_info.update(cam_intrinsic=intrinsic)
                    agent_info['cams'][cam] = cam_info
                timestamp_info['data_dict'][agent_name] = agent_info
            
            elif 'drone' in agent_name.lower():
                agent_type = 'drone'
                for anno in annos['agentInfo']:
                    if anno['agentName'] == agent_name:
                        agent_anno = anno
                        break
                position = agent_anno['localization']['pose']['position']
                orientation = agent_anno['localization']['pose']['orientation']
                ego2global_translation = np.array([position['x'], position['y'], position['z']]).astype(np.float32)
                ego2global_rotation = R.from_quat([orientation['qx'], orientation['qy'], orientation['qz'], orientation['qw']]).as_matrix().astype(np.float32)
                # FRD -> RFU
                ego2global_rotation = ego2global_rotation @ np.array(
                    [[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, -1]]
                ).astype(np.float32)
                agent_pos = np.array(
                    [position['x'], position['y'], position['z']]).astype(np.float32)
                agent_info = {
                    'cams': dict(),
                    'sweeps': [],
                    'ego2global_translation': ego2global_translation,
                    'ego2global_rotation': ego2global_rotation,
                    'agent_type': agent_type,
                    'agent_name': agent_name,
                    'position': agent_pos,
                } 
                for cam in drone_camera_types: 
                    cam_path = osp.join(agent_path, f'{cam}_RGB.jpg')
                    cam_info = dict(image_path=cam_path)
                    cam_info['height'] = agent_anno['camera'][cam]['height']
                    cam_info['width'] = agent_anno['camera'][cam]['width']
                    cam_info['sensor2ego_translation'] = agent_anno['camera'][cam]['translation']
                    cam_info['sensor2ego_rotation'] = agent_anno['camera'][cam]['rotation']
                    intrinsic = agent_anno['camera'][cam]['intrinsic']
                    cam_info.update(cam_intrinsic=intrinsic)
                    agent_info['cams'][cam] = cam_info
                timestamp_info['data_dict'][agent_name] = agent_info

            
        for agent_anno in annos['agentInfo']:
            if 'drone' in agent_anno['agentName'].lower():
                continue
            center = agent_anno['localization']['pose']['center']
            orientation = agent_anno['localization']['pose']['orientation']
            yaw = orientation2euler_angles(orientation)[2]
            gt_box = np.array([
                    center['x'], center['y'], center['z'],
                    agent_anno['length'], agent_anno['width'],
                    agent_anno['height'], yaw
                ]).astype(np.float32)
            agent_category = agent_anno['category']
            if agent_category not in VVSimNameMapping:
                continue
            gt_name = VVSimNameMapping[agent_anno['category']]
            velocity = agent_anno['localization']['pose']['linearVelocity']
            gt_velocity = np.array([velocity['x'], velocity['y']]).astype(np.float32)
            timestamp_info['data_dict']['gt_boxes_global'] = np.append(
                timestamp_info['data_dict']['gt_boxes_global'], [gt_box], axis=0)
            timestamp_info['data_dict']['gt_names'] = np.append(
                timestamp_info['data_dict']['gt_names'], [gt_name], axis=0)
            timestamp_info['data_dict']['gt_velocity_global'] = np.append(
                timestamp_info['data_dict']['gt_velocity_global'], [gt_velocity], axis=0)

        scene_info['data_list'].append(timestamp_info)
    return scene_info


def obtain_sensor2top(s2e_t,
                    s2e_r,
                    l2e_t,
                    l2e_r):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        s2e_t (np.ndarray): Translation from sensor to ego.
        s2e_r (np.ndarray): Rotation from sensor to ego.
        l2e_t (np.ndarray): Translation from Top LiDAR to ego.
        l2e_r (np.ndarray): Rotation from Top LiDAR to ego.
    Returns:
        sensor2top (dict): sensor to top information.
    """
    if not isinstance(s2e_t, np.ndarray):
        s2e_t = np.array(s2e_t).astype(np.float32)
    if not isinstance(s2e_r, np.ndarray):
        s2e_r = np.array(s2e_r).astype(np.float32)
    if not isinstance(l2e_t, np.ndarray):
        l2e_t = np.array(l2e_t).astype(np.float32)
    if not isinstance(l2e_r, np.ndarray):
        l2e_r = np.array(l2e_r).astype(np.float32)
    e2l_r = l2e_r.T
    s2l_r = e2l_r @ s2e_r
    s2l_t = e2l_r @ (s2e_t - l2e_t)
    sensor2top = {
        'sensor2top_rotation': s2l_r,
        'sensor2top_translation': s2l_t
    }
    return sensor2top

def orientation2euler_angles(orientation):
    """Convert orientation to Euler angles in ZXY order.

    Args:
        orientation (dict): Orientation with keys 'qw', 'qx', 'qy', 'qz'.
    """
    qw = orientation['qw']
    qx = orientation['qx']
    qy = orientation['qy']
    qz = orientation['qz']
    roll = np.arctan2(2.0 * (qw * qy - qx * qz),
                   2.0 * (qw**2 + qz**2) - 1.0)
    pitch = np.arcsin(2.0 * (qw * qx + qy * qz))
    yaw = np.arctan2(2.0 * (qw * qz - qx * qy),
                   2.0 * (qw**2 + qy**2) - 1.0)
    return roll, pitch, yaw + np.pi/2
