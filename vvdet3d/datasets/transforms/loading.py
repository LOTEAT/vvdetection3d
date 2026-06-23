# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Union

import open3d as o3d
from PIL import Image
import mmcv
import mmengine
import numpy as np
from typing import Any, Dict
from mmcv.transforms import LoadImageFromFile
from mmcv.transforms.base import BaseTransform
from mmdet.datasets.transforms import LoadAnnotations
from mmengine.fileio import get
import torch
from vvdet3d.registry import TRANSFORMS
from vvdet3d.structures.bbox_3d import get_box_type
from vvdet3d.structures.points import BasePoints, get_points_type


@TRANSFORMS.register_module()
class DistanceFilter(BaseTransform):
    """Filter agents by distance.

    Args:
        "communication_range": float, The communication range.
    """

    def __init__(self,
                 communication_range: float = 70.0) -> None:
        # TODO:
        self.communication_range = communication_range

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        ego_vehicle = results['ego_vehicle']
        agents = results['agents']
        ego_pos = results['position'][ego_vehicle][:2]
        available_agents = []
        unavailable_agents = []
        
        for agent in agents:
            agent_pose = results['position'][agent][:2]
            distance = np.linalg.norm(ego_pos - agent_pose)
            if distance <= self.communication_range:
                available_agents.append(agent)
            else:
                unavailable_agents.append(agent)
        

        results['agents'] = available_agents
        for agent in unavailable_agents:
            results['infos'].pop(agent)

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'communication_range={self.communication_range}, '
        return repr_str




@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles(BaseTransform):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'unchanged',
                 backend_args: Optional[dict] = None,
                 set_default_scale: bool = True) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.backend_args = backend_args
        self.set_default_scale = set_default_scale

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        agents = results['agents']
        for agent in agents:
            filename, cam2img, lidar2cam, cam2img, cam2ego = [], [], [], [], []
            agent_info = results['infos'][agent]
            for _, cam_item in agent_info['images'].items():
                filename.append(cam_item['img_path'])
                cam2img.append(cam_item['cam2img'])
                cam2ego.append(cam_item['cam2ego'])
                if 'drone' in agent.lower():
                    continue
                else:
                    lidar2cam.append(cam_item['lidar2cam'])
            results['infos'][agent]['filename'] = filename
            results['infos'][agent]['cam2img'] = cam2img
            results['infos'][agent]['lidar2cam'] = lidar2cam
            results['infos'][agent]['cam2ego'] = cam2ego
            results['infos'][agent]['ori_cam2img'] = cam2img
            # img is of shape (h, w, c, num_views)
            # h and w can be different for different views
            img_bytes = [
                get(name, backend_args=self.backend_args) for name in filename
            ]
            imgs = [
                mmcv.imfrombytes(img_byte, flag=self.color_type)
                for img_byte in img_bytes
            ]
            # handle the image with different shape
            img_shapes = np.stack([img.shape for img in imgs], axis=0)
            img_shape_max = np.max(img_shapes, axis=0)
            img_shape_min = np.min(img_shapes, axis=0)
            assert img_shape_min[-1] == img_shape_max[-1]
            if not np.all(img_shape_max == img_shape_min):
                pad_shape = img_shape_max[:2]
            else:
                pad_shape = None
            if pad_shape is not None:
                imgs = [
                    mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
                ]
            img = np.stack(imgs, axis=-1)
            if self.to_float32:
                img = img.astype(np.float32)
            # unravel to list, see `DefaultFormatBundle` in formating.py
            # which will transpose each image separately and then stack into array
            results['infos'][agent]['img'] = [img[..., i] for i in range(img.shape[-1])]
            results['infos'][agent]['img_shape'] = img.shape[:2]
            results['infos'][agent]['ori_shape'] = img.shape[:2]
            # Set initial values for default meta_keys
            results['infos'][agent]['pad_shape'] = img.shape[:2]
            if self.set_default_scale:
                results['infos'][agent]['scale_factor'] = 1.0
            num_channels = 1 if len(img.shape) < 3 else img.shape[2]
            results['infos'][agent]['img_norm_cfg'] = dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False)
            results['infos'][agent]['num_views'] = len(filename)
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        return repr_str


@TRANSFORMS.register_module()
class ImageAug3D(BaseTransform):

    def __init__(self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
                 is_train):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        agents = results['agents']
        for agent in agents:
            agent_info = results['infos'][agent]
            imgs = agent_info['img']
            new_imgs = []
            transforms = []
            for img in imgs:
                resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                    agent_info)
                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)
                new_img, rotation, translation = self.img_transform(
                    img,
                    post_rot,
                    post_tran,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                transform = torch.eye(4)
                transform[:2, :2] = rotation
                transform[:2, 2] = translation
                new_imgs.append(np.array(new_img).astype(np.float32))
                transforms.append(transform.numpy())
            results['infos'][agent]['img'] = new_imgs
            # update the calibration matrices
            results['infos'][agent]['img_aug_matrix'] = transforms
        return results




@TRANSFORMS.register_module()
class LoadPointsFromFile(BaseTransform):
    """Load Points From File.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:

            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int] | int): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        norm_intensity (bool): Whether to normlize the intensity. Defaults to
            False.
        norm_elongation (bool): Whether to normlize the elongation. This is
            usually used in Waymo dataset.Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 use_downsample: bool = True,
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        self.shift_height = shift_height
        self.use_downsample = use_downsample
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.backend_args = backend_args

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if pts_filename.endswith('.pcd'):
            pcd = o3d.io.read_point_cloud(pts_filename)
            if self.use_downsample:
                # voxel downsample
                # TODO: modify this hard-coded voxel size
                pcd = pcd.voxel_down_sample(voxel_size=0.1)
            points = np.asarray(pcd.points, dtype=np.float32)
        else:
            try:
                pts_bytes = get(pts_filename, backend_args=self.backend_args)
                points = np.frombuffer(pts_bytes, dtype=np.float32)
            except ConnectionError:
                mmengine.check_file_exist(pts_filename)
                if pts_filename.endswith('.npy'):
                    points = np.load(pts_filename)
                else:
                    points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        agents = results['agents']
        ego_agent_info = results['infos'][results['ego_vehicle']]
        ego_ego2global = np.array(ego_agent_info['ego2global'], dtype=np.float32)
        ego_global2ego = np.linalg.inv(ego_ego2global)
        ego_lidar2ego = np.array(ego_agent_info['lidar_points']['lidar2ego'], dtype=np.float32)
        ego_ego2lidar = np.linalg.inv(ego_lidar2ego)
        
        temp_points_list = []
        for agent in agents:
            if 'drone' in agent.lower():
                continue
            pts_file_path = results['infos'][agent]['lidar_points']['lidar_path']
            points = self._load_points(pts_file_path)
            points = points.reshape(-1, self.load_dim)
            agent_ego2global = np.array(results['infos'][agent]['ego2global'], dtype=np.float32)
            agent_lidar2ego = np.array(results['infos'][agent]['lidar_points']['lidar2ego'], dtype=np.float32)
            results['infos'][agent]['lidar2ego'] = agent_lidar2ego
            agent_lidar2ego_lidar = ego_ego2lidar @ ego_global2ego @ agent_ego2global @ agent_lidar2ego
            
            points_xyz = points[:, :3]
            points_xyz_hom = np.concatenate(
                [points_xyz, np.ones([points_xyz.shape[0], 1])], axis=1
            )
            projected_points_xyz = agent_lidar2ego_lidar @ points_xyz_hom.T
            points = np.concatenate((projected_points_xyz[:3, :].T, points[:, 3:]), axis=1)
            points = points[:, self.use_dim]
            temp_points_list.append(points)
            if self.norm_intensity:
                assert len(self.use_dim) >= 4, \
                    f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
                points[:, 3] = np.tanh(points[:, 3])
            if self.norm_elongation:
                assert len(self.use_dim) >= 5, \
                    f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
                points[:, 4] = np.tanh(points[:, 4])
            attribute_dims = None
            if self.shift_height:
                floor_height = np.percentile(points[:, 2], 0.99)
                height = points[:, 2] - floor_height
                points = np.concatenate(
                    [points[:, :3],
                    np.expand_dims(height, 1), points[:, 3:]], 1)
                attribute_dims = dict(height=3)
            if self.use_color:
                assert len(self.use_dim) >= 6
                if attribute_dims is None:
                    attribute_dims = dict()
                attribute_dims.update(
                    dict(color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]))
            points_class = get_points_type(self.coord_type)
            points = points_class(
                points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
            results['infos'][agent]['points'] = points
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        repr_str += f'norm_intensity={self.norm_intensity})'
        repr_str += f'norm_elongation={self.norm_elongation})'
        return repr_str


@TRANSFORMS.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def transform(self, results: dict) -> dict:
        """Convert the type of points from ndarray to corresponding
        `point_class`.

        Args:
            results (dict): input result. The value of key `points` is a
                numpy array.

        Returns:
            dict: The processed results.
        """
        assert 'points' in results
        points = results['points']

        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        return results


@TRANSFORMS.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Required Keys:

    - ann_info (dict)

        - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes` |
          :obj:`DepthInstance3DBoxes` | :obj:`CameraInstance3DBoxes`):
          3D ground truth bboxes. Only when `with_bbox_3d` is True
        - gt_labels_3d (np.int64): Labels of ground truths.
          Only when `with_label_3d` is True.
        - gt_bboxes (np.float32): 2D ground truth bboxes.
          Only when `with_bbox` is True.
        - gt_labels (np.ndarray): Labels of ground truths.
          Only when `with_label` is True.
        - depths (np.ndarray): Only when
          `with_bbox_depth` is True.
        - centers_2d (np.ndarray): Only when
          `with_bbox_depth` is True.
        - attr_labels (np.ndarray): Attribute labels of instances.
          Only when `with_attr_label` is True.

    - pts_instance_mask_path (str): Path of instance mask file.
      Only when `with_mask_3d` is True.
    - pts_semantic_mask_path (str): Path of semantic mask file.
      Only when `with_seg_3d` is True.
    - pts_panoptic_mask_path (str): Path of panoptic mask file.
      Only when both `with_panoptic_3d` is True.

    Added Keys:

    - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes` |
      :obj:`DepthInstance3DBoxes` | :obj:`CameraInstance3DBoxes`):
      3D ground truth bboxes. Only when `with_bbox_3d` is True
    - gt_labels_3d (np.int64): Labels of ground truths.
      Only when `with_label_3d` is True.
    - gt_bboxes (np.float32): 2D ground truth bboxes.
      Only when `with_bbox` is True.
    - gt_labels (np.int64): Labels of ground truths.
      Only when `with_label` is True.
    - depths (np.float32): Only when
      `with_bbox_depth` is True.
    - centers_2d (np.ndarray): Only when
      `with_bbox_depth` is True.
    - attr_labels (np.int64): Attribute labels of instances.
      Only when `with_attr_label` is True.
    - pts_instance_mask (np.int64): Instance mask of each point.
      Only when `with_mask_3d` is True.
    - pts_semantic_mask (np.int64): Semantic mask of each point.
      Only when `with_seg_3d` is True.

    Args:
        with_bbox_3d (bool): Whether to load 3D boxes. Defaults to True.
        with_label_3d (bool): Whether to load 3D labels. Defaults to True.
        with_attr_label (bool): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool): Whether to load 3D instance masks for points.
            Defaults to False.
        with_seg_3d (bool): Whether to load 3D semantic masks for points.
            Defaults to False.
        with_bbox (bool): Whether to load 2D boxes. Defaults to False.
        with_label (bool): Whether to load 2D labels. Defaults to False.
        with_mask (bool): Whether to load 2D instance masks. Defaults to False.
        with_seg (bool): Whether to load 2D semantic masks. Defaults to False.
        with_bbox_depth (bool): Whether to load 2.5D boxes. Defaults to False.
        with_panoptic_3d (bool): Whether to load 3D panoptic masks for points.
            Defaults to False.
        poly2mask (bool): Whether to convert polygon annotations to bitmasks.
            Defaults to True.
        seg_3d_dtype (str): String of dtype of 3D semantic masks.
            Defaults to 'np.int64'.
        seg_offset (int): The offset to split semantic and instance labels from
            panoptic labels. Defaults to None.
        dataset_type (str): Type of dataset used for splitting semantic and
            instance labels. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 with_bbox_3d: bool = True,
                 with_label_3d: bool = True,
                 with_attr_label: bool = False,
                 with_mask_3d: bool = False,
                 with_seg_3d: bool = False,
                 with_bbox: bool = False,
                 with_label: bool = False,
                 with_mask: bool = False,
                 with_seg: bool = False,
                 with_bbox_depth: bool = False,
                 with_panoptic_3d: bool = False,
                 poly2mask: bool = True,
                 seg_3d_dtype: str = 'np.int64',
                 seg_offset: int = None,
                 dataset_type: str = None,
                 backend_args: Optional[dict] = None) -> None:
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask,
            backend_args=backend_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.with_panoptic_3d = with_panoptic_3d
        self.seg_3d_dtype = eval(seg_3d_dtype)
        self.seg_offset = seg_offset
        self.dataset_type = dataset_type

    def _load_bboxes_3d(self, results: dict) -> dict:
        """Private function to move the 3D bounding box annotation from
        `ann_info` field to the root of `results`.

        Args:
            results (dict): Result dict from :obj:`vvdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        return results

    def _load_bboxes_depth(self, results: dict) -> dict:
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`vvdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """

        results['depths'] = results['ann_info']['depths']
        results['centers_2d'] = results['ann_info']['centers_2d']
        return results

    def _load_labels_3d(self, results: dict) -> dict:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`vvdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """

        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results: dict) -> dict:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`vvdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results: dict) -> dict:
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`vvdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['pts_instance_mask_path']

        try:
            mask_bytes = get(
                pts_instance_mask_path, backend_args=self.backend_args)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmengine.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
        return results

    def _load_semantic_seg_3d(self, results: dict) -> dict:
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`vvdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['pts_semantic_mask_path']

        try:
            mask_bytes = get(
                pts_semantic_mask_path, backend_args=self.backend_args)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmengine.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        if self.dataset_type == 'semantickitti':
            pts_semantic_mask = pts_semantic_mask.astype(np.int64)
            pts_semantic_mask = pts_semantic_mask % self.seg_offset
        # nuScenes loads semantic and panoptic labels from different files.

        results['pts_semantic_mask'] = pts_semantic_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
        return results

    def _load_panoptic_3d(self, results: dict) -> dict:
        """Private function to load 3D panoptic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`vvdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the panoptic segmentation annotations.
        """
        pts_panoptic_mask_path = results['pts_panoptic_mask_path']

        try:
            mask_bytes = get(
                pts_panoptic_mask_path, backend_args=self.backend_args)
            # add .copy() to fix read-only bug
            pts_panoptic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmengine.check_file_exist(pts_panoptic_mask_path)
            pts_panoptic_mask = np.fromfile(
                pts_panoptic_mask_path, dtype=np.int64)

        if self.dataset_type == 'semantickitti':
            pts_semantic_mask = pts_panoptic_mask.astype(np.int64)
            pts_semantic_mask = pts_semantic_mask % self.seg_offset
        elif self.dataset_type == 'nuscenes':
            pts_semantic_mask = pts_semantic_mask // self.seg_offset

        results['pts_semantic_mask'] = pts_semantic_mask

        # We can directly take panoptic labels as instance ids.
        pts_instance_mask = pts_panoptic_mask.astype(np.int64)
        results['pts_instance_mask'] = pts_instance_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
            results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
        return results

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        The only difference is it remove the proceess for
        `ignore_flag`

        Args:
            results (dict): Result dict from :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        results['gt_bboxes'] = results['ann_info']['gt_bboxes']

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        results['gt_bboxes_labels'] = results['ann_info']['gt_bboxes_labels']

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`vvdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
            semantic segmentation annotations.
        """
        results = super().transform(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_panoptic_3d:
            results = self._load_panoptic_3d(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_panoptic_3d={self.with_panoptic_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        repr_str += f'{indent_str}seg_offset={self.seg_offset})'

        return repr_str


@TRANSFORMS.register_module()
class LidarDet3DInferencerLoader(BaseTransform):
    """Load point cloud in the Inferencer's pipeline.

    Added keys:
      - points
      - timestamp
      - axis_align_matrix
      - box_type_3d
      - box_mode_3d
    """

    def __init__(self, coord_type='LIDAR', **kwargs) -> None:
        super().__init__()
        self.from_file = TRANSFORMS.build(
            dict(type='LoadPointsFromFile', coord_type=coord_type, **kwargs))
        self.from_ndarray = TRANSFORMS.build(
            dict(type='LoadPointsFromDict', coord_type=coord_type, **kwargs))
        self.box_type_3d, self.box_mode_3d = get_box_type(coord_type)

    def transform(self, single_input: dict) -> dict:
        """Transform function to add image meta information.
        Args:
            single_input (dict): Single input.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert 'points' in single_input, "key 'points' must be in input dict"
        if isinstance(single_input['points'], str):
            inputs = dict(
                lidar_points=dict(lidar_path=single_input['points']),
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        elif isinstance(single_input['points'], np.ndarray):
            inputs = dict(
                points=single_input['points'],
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        else:
            raise ValueError('Unsupported input points type: '
                             f"{type(single_input['points'])}")

        if 'points' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)
    
    



@TRANSFORMS.register_module()
class CoDet3DInferencerPointsLoader(BaseTransform):
    """Load Points From File.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:

            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int] | int): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        norm_intensity (bool): Whether to normlize the intensity. Defaults to
            False.
        norm_elongation (bool): Whether to normlize the elongation. This is
            usually used in Waymo dataset.Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 use_downsample: bool = True,
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 save_path: str = None,
                 backend_args: Optional[dict] = None) -> None:
        self.shift_height = shift_height
        self.use_downsample = use_downsample
        self.use_color = use_color
        self.save_path = save_path
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.backend_args = backend_args

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if pts_filename.endswith('.pcd'):
            pcd = o3d.io.read_point_cloud(pts_filename)
            if self.use_downsample:
                # voxel downsample
                # TODO: modify this hard-coded voxel size
                pcd = pcd.voxel_down_sample(voxel_size=0.1)
            points = np.asarray(pcd.points, dtype=np.float32)
        else:
            try:
                pts_bytes = get(pts_filename, backend_args=self.backend_args)
                points = np.frombuffer(pts_bytes, dtype=np.float32)
            except ConnectionError:
                mmengine.check_file_exist(pts_filename)
                if pts_filename.endswith('.npy'):
                    points = np.load(pts_filename)
                else:
                    points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        agents = results['agents']
        ego_agent_info = results['infos'][results['ego_vehicle']]
        ego_ego2global = np.array(ego_agent_info['ego2global'], dtype=np.float32)
        ego_global2ego = np.linalg.inv(ego_ego2global)
        ego_lidar2ego = np.array(ego_agent_info['lidar_points']['lidar2ego'], dtype=np.float32)
        ego_ego2lidar = np.linalg.inv(ego_lidar2ego)
        
        combined_points = []
        for agent in agents:
            if 'drone' in agent.lower():
                continue
            pts_file_path = results['infos'][agent]['lidar_points']['lidar_path']
            print(pts_file_path)
            points = self._load_points(pts_file_path)
            points = points.reshape(-1, self.load_dim)
            agent_ego2global = np.array(results['infos'][agent]['ego2global'], dtype=np.float32)
            agent_lidar2ego = np.array(results['infos'][agent]['lidar_points']['lidar2ego'], dtype=np.float32)
            results['infos'][agent]['lidar2ego'] = agent_lidar2ego
            agent_lidar2ego_lidar = ego_ego2lidar @ ego_global2ego @ agent_ego2global @ agent_lidar2ego
            
            points_xyz = points[:, :3]
            points_xyz_hom = np.concatenate(
                [points_xyz, np.ones([points_xyz.shape[0], 1])], axis=1
            )
            projected_points_xyz = agent_lidar2ego_lidar @ points_xyz_hom.T
            points = np.concatenate((projected_points_xyz[:3, :].T, points[:, 3:]), axis=1)
            points = points[:, self.use_dim]
            combined_points.append(points)
            if self.norm_intensity:
                assert len(self.use_dim) >= 4, \
                    f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
                points[:, 3] = np.tanh(points[:, 3])
            if self.norm_elongation:
                assert len(self.use_dim) >= 5, \
                    f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
                points[:, 4] = np.tanh(points[:, 4])
            attribute_dims = None
            if self.shift_height:
                floor_height = np.percentile(points[:, 2], 0.99)
                height = points[:, 2] - floor_height
                points = np.concatenate(
                    [points[:, :3],
                    np.expand_dims(height, 1), points[:, 3:]], 1)
                attribute_dims = dict(height=3)
            if self.use_color:
                assert len(self.use_dim) >= 6
                if attribute_dims is None:
                    attribute_dims = dict()
                attribute_dims.update(
                    dict(color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]))
            points_class = get_points_type(self.coord_type)
            points = points_class(
                points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
            results['infos'][agent]['points'] = points
        combined_points = np.vstack(combined_points, dtype=np.float32)
        combined_points.tofile(self.save_path)
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        repr_str += f'norm_intensity={self.norm_intensity})'
        repr_str += f'norm_elongation={self.norm_elongation})'
        return repr_str


@TRANSFORMS.register_module()
class MonoDet3DInferencerLoader(BaseTransform):
    """Load an image from ``results['images']['CAMX']['img']``. Similar with
    :obj:`LoadImageFromFileMono3D`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['images']['CAMX']['img']``.

    Added keys:
      - img
      - box_type_3d
      - box_mode_3d

    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.from_file = TRANSFORMS.build(
            dict(type='LoadImageFromFileMono3D', **kwargs))
        self.from_ndarray = TRANSFORMS.build(
            dict(type='LoadImageFromNDArray', **kwargs))

    def transform(self, single_input: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            single_input (dict): Result dict with Webcam read image in
                ``results['images']['CAMX']['img']``.
        Returns:
            dict: The dict contains loaded image and meta information.
        """
        box_type_3d, box_mode_3d = get_box_type('camera')

        if isinstance(single_input['img'], str):
            inputs = dict(
                images=dict(
                    CAM_FRONT=dict(
                        img_path=single_input['img'],
                        cam2img=single_input['cam2img'])),
                box_mode_3d=box_mode_3d,
                box_type_3d=box_type_3d)
        elif isinstance(single_input['img'], np.ndarray):
            inputs = dict(
                img=single_input['img'],
                cam2img=single_input['cam2img'],
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)
        else:
            raise ValueError('Unsupported input image type: '
                             f"{type(single_input['img'])}")

        if 'img' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)


@TRANSFORMS.register_module()
class MultiModalityDet3DInferencerLoader(BaseTransform):
    """Load point cloud and image in the Inferencer's pipeline.

    Added keys:
      - points
      - img
      - cam2img
      - lidar2cam
      - lidar2img
      - timestamp
      - axis_align_matrix
      - box_type_3d
      - box_mode_3d
    """

    def __init__(self, load_point_args: dict, load_img_args: dict) -> None:
        super().__init__()
        self.points_from_file = TRANSFORMS.build(
            dict(type='LoadPointsFromFile', **load_point_args))
        self.points_from_ndarray = TRANSFORMS.build(
            dict(type='LoadPointsFromDict', **load_point_args))
        coord_type = load_point_args['coord_type']
        self.box_type_3d, self.box_mode_3d = get_box_type(coord_type)

        self.imgs_from_file = TRANSFORMS.build(
            dict(type='LoadImageFromFile', **load_img_args))
        self.imgs_from_ndarray = TRANSFORMS.build(
            dict(type='LoadImageFromNDArray', **load_img_args))

    def transform(self, single_input: dict) -> dict:
        """Transform function to add image meta information.
        Args:
            single_input (dict): Single input.

        Returns:
            dict: The dict contains loaded image, point cloud and meta
            information.
        """
        assert 'points' in single_input and 'img' in single_input, \
            "key 'points', 'img' and must be in input dict," \
            f'but got {single_input}'
        if isinstance(single_input['points'], str):
            inputs = dict(
                lidar_points=dict(lidar_path=single_input['points']),
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        elif isinstance(single_input['points'], np.ndarray):
            inputs = dict(
                points=single_input['points'],
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)
        else:
            raise ValueError('Unsupported input points type: '
                             f"{type(single_input['points'])}")

        if 'points' in inputs:
            points_inputs = self.points_from_ndarray(inputs)
        else:
            points_inputs = self.points_from_file(inputs)

        multi_modality_inputs = points_inputs

        box_type_3d, box_mode_3d = get_box_type('lidar')

        if isinstance(single_input['img'], str):
            inputs = dict(
                img_path=single_input['img'],
                cam2img=single_input['cam2img'],
                lidar2img=single_input['lidar2img'],
                lidar2cam=single_input['lidar2cam'],
                box_mode_3d=box_mode_3d,
                box_type_3d=box_type_3d)
        elif isinstance(single_input['img'], np.ndarray):
            inputs = dict(
                img=single_input['img'],
                cam2img=single_input['cam2img'],
                lidar2img=single_input['lidar2img'],
                lidar2cam=single_input['lidar2cam'],
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)
        else:
            raise ValueError('Unsupported input image type: '
                             f"{type(single_input['img'])}")

        if isinstance(single_input['img'], np.ndarray):
            imgs_inputs = self.imgs_from_ndarray(inputs)
        else:
            imgs_inputs = self.imgs_from_file(inputs)

        multi_modality_inputs.update(imgs_inputs)

        return multi_modality_inputs
