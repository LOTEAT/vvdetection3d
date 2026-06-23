import argparse
import glob
from concurrent.futures import ThreadPoolExecutor
from os import path as osp

import numpy as np
import open3d as o3d
from tqdm import tqdm

from tools.dataset_converters import v2xset_converter as v2xset_converter
from tools.dataset_converters import vvsim_converter as vvsim_converter
from tools.dataset_converters.update_infos import update_pkl_infos


def convert_frd_to_flu(pcd_folder, workers):
    if not osp.isdir(pcd_folder):
        raise FileNotFoundError(f'No such directory: {pcd_folder}')

    def process_pcd(file_path):
        new_file_path = file_path.replace('.pcd', '.pcd.bin')
        if osp.exists(new_file_path):
            return
        pcd = o3d.io.read_point_cloud(file_path)
        voxel_size = 0.1
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        points = np.asarray(downsampled_pcd.points, dtype=np.float32)
        points[:, 1] = -points[:, 1]
        points[:, 2] = -points[:, 2]
        points.tofile(new_file_path)

    pcd_files = glob.glob(
        osp.join(pcd_folder, '**', 'LIDAR_TOP.pcd'), recursive=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for _ in tqdm(
                executor.map(process_pcd, pcd_files),
                total=len(pcd_files),
                desc=f'Processing PCD files in {pcd_folder}'):
            pass


def vvsim_data_prep(root_path,
                    info_prefix,
                    data_split,
                    dataset_name,
                    out_dir,
                    max_sweeps,
                    workers):
    """Prepare data related to VVSim dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        data_split (str): The split of dataset, e.g., 'train', 'val', 'test', 'trainval'.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
        workers (int): Number of threads to be used.
    """
    if data_split == 'trainval':
        for split in ['train', 'val']:
            convert_frd_to_flu(osp.join(root_path, split), workers)
    else:
        convert_frd_to_flu(osp.join(root_path, data_split), workers)
    vvsim_converter.create_vvsim_infos(
        root_path, info_prefix, data_split=data_split, max_sweeps=max_sweeps, workers=workers)
    info_path = osp.join(out_dir, f'{info_prefix}_infos_{data_split}.pkl')
    update_pkl_infos('vvsim', out_dir=out_dir, pkl_path=info_path)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='v2xset', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/vvsim',
    help='specify the root path of dataset')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/vvsim',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='vvsim')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    from mmengine.registry import init_default_scope
    init_default_scope('vvdet3d')

    if args.dataset == "vvsim":
        for data_split in ['train', 'trainval', 'val', 'test']:
            vvsim_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                data_split=data_split,
                dataset_name='VVSimDataset',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps,
                workers=args.workers)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
