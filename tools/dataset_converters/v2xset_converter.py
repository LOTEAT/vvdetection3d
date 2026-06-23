import os
from os import path as osp
import mmengine
from mmengine import track_iter_progress, track_parallel_progress

v2xset_categories = ('car',)

def create_v2xset_infos(root_path,
                        info_prefix,
                        data_split='train',
                        max_sweeps=10,
                        workers=4):
    """Create info file of v2xset dataset.

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
    available_data_split = ['train', 'val', 'test', 'trainval']

    def get_scenes(data_split):
        folder_path = osp.join(root_path, data_split)
        assert mmengine.isdir(
            folder_path
        ), f'FileNotFoundError: No such directory: \'{data_split}\''
        return sorted([
            osp.join(folder_path, scene) for scene in os.listdir(folder_path)
        ])
        
    assert data_split in available_data_split
    if data_split == 'trainval':
        scenes = get_scenes('train') + get_scenes('validate')
    elif data_split == 'val':
        scenes = get_scenes('validate')
    else:
        scenes = get_scenes(data_split)
    print('{} scenes: {}'.format(data_split, len(scenes)))
    v2xset_data_infos = []
    if workers == 0:
        for scene in track_iter_progress(scenes):
            v2xset_data_infos.append(_fill_infos(scene))
    else:
        v2xset_data_infos = track_parallel_progress(_fill_infos, scenes, nproc=workers)
    metadata = dict()
    data = dict(infos=v2xset_data_infos, metadata=metadata)
    info_path = osp.join(root_path,
                        '{}_infos_{}.pkl'.format(info_prefix, data_split))
    mmengine.dump(data, info_path)


def _fill_infos(scene):
    """Generate the train/val/test/trainval infos from the raw data.

    Args:
        scenes (list[str]): Basic information of scenes.

    Returns:
        OrderedDict: Information of train/val/test/trainval set,
        that will be saved to the info file.
    """
    camera_types = ['camera0', 'camera1', 'camera2', 'camera3']
    agents = sorted([
        agent for agent in os.listdir(scene)
        if mmengine.isdir(os.path.join(scene, agent))
    ])
    # RSU's id is always negative, make sure they will be in the end of
    # the list as they shouldn't be ego vehicle.
    if int(agents[0]) < 0:
        agents = agents[1:] + [agents[0], ]
    agent_path = osp.join(scene, agents[0]) 
    files = os.listdir(agent_path)
    yaml_files = [f for f in files if f.endswith(".yaml") and "additional" not in f]
    info_files = sorted(os.path.join(agent_path, f) for f in yaml_files)
    timestamps = [
        osp.splitext(osp.split(info_file)[-1])[0]
        for info_file in info_files
    ]
    metainfo = {
        'scene_name': osp.split(scene)[-1],
        'num_agents': len(agents),
        'scene_length': len(timestamps)
    }
    scene_info = {
        'data_list' : [],
        'metainfo': metainfo
    }
    for timestamp in timestamps:
        timestamp_info = {
            'metadata': dict(),
            'data_dict': dict()
        }
        for agent_id in agents:
            agent_path = osp.join(scene, agent_id)
            info_path = osp.join(agent_path, f'{timestamp}.yaml')
            lidar_path = osp.join(agent_path, f'{timestamp}.pcd')
            agent_info = {
                'lidar_path': lidar_path,
                'info_path': info_path,
                'cams': dict(),
                'sweeps': [],
                'num_features': 4,
                'is_ego': False,
                'agent_type': 'car',
                'agent_id': agent_id
            } 
            for cam in camera_types: 
                cam_path = osp.join(agent_path, f'{timestamp}_{cam}.png')
                if osp.isfile(cam_path):
                    agent_info['cams'][cam] = dict(image_path=cam_path)
            if agent_id == agents[0]:
                agent_info['is_ego'] = True
            if int(agent_id) < 0:
                agent_info['agent_type'] = 'rsu'
            timestamp_info['data_dict'][agent_id] = agent_info
            timestamp_info['metadata'] = {
                'timestamp': timestamp,
                'ego_agent_id': agents[0]
            }
        scene_info['data_list'].append(timestamp_info)
    return scene_info


