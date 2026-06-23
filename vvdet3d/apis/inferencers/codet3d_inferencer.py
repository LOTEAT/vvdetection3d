# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Sequence, Union
import os
import mmengine
import numpy as np
import torch
import json
import cv2
from mmengine.dataset import Compose
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)
from mmengine.infer.infer import ModelType
from mmengine.structures import InstanceData

from vvdet3d.registry import INFERENCERS, DATASETS
from vvdet3d.structures import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                                Det3DDataSample, LiDARInstance3DBoxes)
from vvdet3d.utils import ConfigType
from .base_3d_inferencer import Base3DInferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


def create_bbox_corner_points(box):
    x, y, z, l, w, h, yaw = box
    x_corners = l / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    corners = np.vstack([x_corners, y_corners, z_corners]).T

    R = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [ 0,           0,           1]
    ])
    corners = corners @ R.T
    corners += np.array([x, y, z])
    return corners  # (8,3)



@INFERENCERS.register_module(name='codet3d')
@INFERENCERS.register_module()
class CoDet3DInferencer(Base3DInferencer):
    """The inferencer of LiDAR-based detection.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "pointpillars_kitti-3class" or
            "configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py". # noqa: E501
            If model is not specified, user must provide the
            `weights` saved by MMEngine which contains the config string.
            Defaults to None.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str): The scope of the model. Defaults to 'vvdet3d'.
        palette (str): Color palette used for visualization. The order of
            priority is palette -> config -> checkpoint. Defaults to 'none'.
    """

    def __init__(self,
                 dataloader: str,
                 sample_idx: int,
                 out_dir: str,
                 model: Union[ModelType, str, None] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: str = 'vvdet3d',
                 palette: str = 'none') -> None:
        # A global counter tracking the number of frames processed, for
        # naming of the output results
        self.dataloader = dataloader
        self.sample_idx = sample_idx
        self.num_visualized_frames = 0
        self.num_visualized_imgs = 0
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        super(CoDet3DInferencer, self).__init__(
            model=model,
            weights=weights,
            device=device,
            scope=scope,
            palette=palette)
        dataset = self.cfg.get(f'{dataloader}_dataloader').dataset
        self.info_path = dataset.data_root + dataset.ann_file
        self.ego_vehicle = dataset.ego_vehicle
        print("Building dataset from", f'{dataloader}_dataloader')
        dataset_pipeline = dataset.pipeline
        load_point_idx = self._get_transform_idx(dataset_pipeline,
                                                 'LoadPointsFromFile')
        if load_point_idx == -1:
            raise ValueError(
                'LoadPointsFromFile is not found in the test pipeline')

        load_cfg = dataset_pipeline[load_point_idx]
        self.coord_type, self.load_dim = load_cfg['coord_type'], load_cfg[
            'load_dim']
        self.use_dim = list(range(load_cfg['use_dim'])) if isinstance(
            load_cfg['use_dim'], int) else load_cfg['use_dim']
        load_cfg['save_path'] = osp.join(self.out_dir, 'combined_points.pcd.bin')

        load_cfg['type'] = 'CoDet3DInferencerPointsLoader'
        self.dataset = DATASETS.build(dataset)
    
    def prepare_data(self):
        """Prepare data for inference."""
        data_item = self.dataset[self.sample_idx]
        return data_item


    def _inputs_to_list(self, inputs: Union[dict, list], **kwargs) -> list:
        if isinstance(inputs, str):
            backend = get_file_backend(inputs)
            if hasattr(backend, 'isdir') and isdir(inputs):
                # Backends like HttpsBackend do not implement `isdir`, so only
                # those backends that implement `isdir` could accept the inputs
                # as a directory
                filename_list = list_dir_or_file(inputs, list_dir=False)
                inputs = [
                    join_path(inputs, filename) for filename in filename_list
                ]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        load_point_idx = self._get_transform_idx(pipeline_cfg,
                                                 'LoadPointsFromFile')
        if load_point_idx == -1:
            raise ValueError(
                'LoadPointsFromFile is not found in the test pipeline')

        load_cfg = pipeline_cfg[load_point_idx]
        self.coord_type, self.load_dim = load_cfg['coord_type'], load_cfg[
            'load_dim']
        self.use_dim = list(range(load_cfg['use_dim'])) if isinstance(
            load_cfg['use_dim'], int) else load_cfg['use_dim']
        return Compose(pipeline_cfg)

    def preprocess(self, inputs: InputsType):
        return self.collate_fn(inputs)
    


    def __call__(self,
                 inputs: InputsType,
                 return_datasamples: bool = False,
                 **kwargs) -> Optional[dict]:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            batch_size (int): Batch size. Defaults to 1.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.


        Returns:
            dict: Inference and visualization results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        cam_type = preprocess_kwargs.pop('cam_type', 'CAM2')
        ori_inputs = self._inputs_to_list(inputs, cam_type=cam_type)
        inputs = self.preprocess(ori_inputs)
        preds = []

        results_dict = {'predictions': [], 'visualization': []}
        preds.extend(self.forward(inputs, **forward_kwargs))
        visualization = self.visualize(ori_inputs, preds,
                                        **visualize_kwargs)
        results = self.postprocess(preds, visualization,
                                    return_datasamples,
                                    **postprocess_kwargs)
        results_dict['predictions'].extend(results['predictions'])
        if results['visualization'] is not None:
            results_dict['visualization'].extend(results['visualization'])
        self.visualize_imgs(visualize_kwargs['pred_score_thr'])
        return results_dict

    def visualize_imgs(self, pred_score_thr):
        vis_imgs_dir = osp.join(self.out_dir, 'vis_imgs')
        os.makedirs(vis_imgs_dir, exist_ok=True)
        infos = mmengine.load(self.info_path)
        scene_len = [scene_data['metainfo']['scene_length'] for scene_data in infos['scene_list']]
        scene_cumlen = np.cumsum([0] + scene_len).tolist()
        sample_idx = self.sample_idx
        scene_idx = None
        
        for i in range(len(scene_cumlen) - 1):
            if scene_cumlen[i] <= sample_idx < scene_cumlen[i + 1]:
                scene_idx = i
                break
        frame_idx_in_scene = sample_idx - scene_cumlen[scene_idx]
        data_dict = infos['scene_list'][scene_idx]['data_list'][frame_idx_in_scene]['data_dict']
        lidar2ego = np.array(data_dict[self.ego_vehicle]['lidar_points']['lidar2ego'], dtype=np.float32)  # 4x4
        ego2global = np.array(data_dict[self.ego_vehicle]['ego2global'], dtype=np.float32)  # 4x4
        lidar2global = ego2global @ lidar2ego
        agents = infos['scene_list'][scene_idx]['data_list'][frame_idx_in_scene]['metainfo']['agents']
        drone_agents = [agent for agent in agents if 'drone' in agent.lower()]
        out_json_path = osp.join(self.out_dir, 'preds',
                    f'{str(self.num_visualized_imgs).zfill(8)}.json')
        with open(out_json_path, 'r') as f:
            pred_json = json.load(f)
        scores = np.array(pred_json['scores_3d'])
        valid_score_mask = scores > pred_score_thr
        gt_boxes = np.array(pred_json['bboxes_3d'])[valid_score_mask, :7]
        for drone_agent in drone_agents:
            drone2global = np.array(data_dict[drone_agent]['ego2global'], dtype=np.float32)  # 4x4
            global2drone = np.linalg.inv(drone2global)
            cam_info = data_dict[drone_agent]['images']['CAM_BOTTOM']
            img_path = cam_info['img_path']
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read image at {img_path}")
            
            height, width = cam_info['height'], cam_info['width']
            cam2img = np.array(cam_info['cam2img'])      # 3x4
            cam2ego = np.array(cam_info['cam2ego'])      # 4x4
            
    
        
            for box in gt_boxes:
                corners_local = create_bbox_corner_points(box)  # 只需要8个角点
                M = corners_local.shape[0]
                hom = np.hstack([corners_local, np.ones((M,1))]).T  # 4x8

                # 坐标变换到相机系
                global_points = lidar2global @ hom         # 4x8
                cam_points = np.linalg.inv(cam2ego) @ global2drone @ global_points   # 4x8
                z = cam_points[2, :]
                valid_mask = z > 0
                if not np.all(valid_mask):
                    continue 

                img_points_h = cam2img @ cam_points[:3, :]  # 3x8
                img_points = img_points_h[:2, :] / img_points_h[2, :]  # 归一化

                img_points = img_points.T  # 8x2


                # 过滤完全在图像外的box
                if np.all((img_points[:,0] < 0) | (img_points[:,0] >= width)) or \
                np.all((img_points[:,1] < 0) | (img_points[:,1] >= height)):
                    continue

                x_min, y_min = np.min(img_points, axis=0)
                x_max, y_max = np.max(img_points, axis=0)
                cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0,255,0), thickness=2)

            cv2.imwrite(osp.join(vis_imgs_dir, f'{drone_agent}.jpg'), img)


    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = -1,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  no_save_vis: bool = False,
                  img_out_dir: str = '') -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            preds (PredType): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to -1.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            no_save_vis (bool): Whether to force not to save prediction
                vis results. Defaults to False.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if no_save_vis is True:
            img_out_dir = ''

        if not show and img_out_dir == '' and not return_vis:
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            combined_points_path = osp.join(self.out_dir, 'combined_points.pcd.bin')
            
            pts_bytes = mmengine.fileio.get(combined_points_path)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
            points = points.reshape(-1, self.load_dim)
            points = points[:, self.use_dim]
            pc_name = osp.basename(combined_points_path).split('.bin')[0]
            pc_name = f'{pc_name}.png'

            if img_out_dir != '' and show:
                o3d_save_path = osp.join(img_out_dir, 'vis_lidar', pc_name)
                mmengine.mkdir_or_exist(osp.dirname(o3d_save_path))
            else:
                o3d_save_path = None

            data_input = dict(points=points)
            self.visualizer.add_datasample(
                pc_name,
                data_input,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=True,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                o3d_save_path=o3d_save_path,
                vis_task='lidar_det',
            )
            results.append(points)
            self.num_visualized_frames += 1

        return results

    def visualize_preds_fromfile(self, inputs: InputsType, preds: PredType,
                                 **kwargs) -> Union[List[np.ndarray], None]:
        """Visualize predictions from `*.json` files.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            preds (PredType): Predictions of the model.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        data_samples = []
        for pred in preds:
            pred = mmengine.load(pred)
            data_sample = Det3DDataSample()
            data_sample.pred_instances_3d = InstanceData()

            data_sample.pred_instances_3d.labels_3d = torch.tensor(
                pred['labels_3d'])
            data_sample.pred_instances_3d.scores_3d = torch.tensor(
                pred['scores_3d'])
            if pred['box_type_3d'] == 'LiDAR':
                data_sample.pred_instances_3d.bboxes_3d = \
                    LiDARInstance3DBoxes(pred['bboxes_3d'])
            elif pred['box_type_3d'] == 'Camera':
                data_sample.pred_instances_3d.bboxes_3d = \
                    CameraInstance3DBoxes(pred['bboxes_3d'])
            elif pred['box_type_3d'] == 'Depth':
                data_sample.pred_instances_3d.bboxes_3d = \
                    DepthInstance3DBoxes(pred['bboxes_3d'])
            else:
                raise ValueError('Unsupported box type: '
                                 f'{pred["box_type_3d"]}')
            data_samples.append(data_sample)
        return self.visualize(inputs=inputs, preds=data_samples, **kwargs)
