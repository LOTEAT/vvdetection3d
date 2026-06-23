# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Sequence

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from vvdet3d.registry import MODELS
from vvdet3d.structures import Det3DDataSample
from .base import Base3DDetector


@MODELS.register_module()
class VVMVXTwoStageDetector(Base3DDetector):
    """Base class of Vehicle-coorperation Multi-modality VoxelNet.

    Args:
        cav_pts_voxel_encoder (dict, optional): Point voxelization
            encoder layer for CAV. Defaults to None.
        cav_pts_middle_encoder (dict, optional): Middle encoder layer
            of points cloud modality for CAV. Defaults to None.
        cav_pts_fusion_layer (dict, optional): Fusion layer.
            Defaults to None.
        cav_img_backbone (dict, optional): Backbone of extracting
            images feature. Defaults to None.
        cav_pts_backbone (dict, optional): Backbone of extracting
            points features. Defaults to None.
        cav_img_neck (dict, optional): Neck of extracting
            image features. Defaults to None.
        cav_pts_neck (dict, optional): Neck of extracting
            points features. Defaults to None.
        bbox_head (dict, optional): Bboxes head of
            point cloud modality. Defaults to None.
        cav_img_roi_head (dict, optional): RoI head of image
            modality. Defaults to None.
        cav_img_rpn_head (dict, optional): RPN head of image
            modality. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`Det3DDataPreprocessor`. Defaults to None.
    """

    def __init__(self,
                 cav_pts_voxel_encoder: Optional[dict] = None,
                 cav_pts_middle_encoder: Optional[dict] = None,
                 cav_pts_fusion_layer: Optional[dict] = None,
                 cav_img_backbone: Optional[dict] = None,
                 cav_pts_backbone: Optional[dict] = None,
                 cav_img_neck: Optional[dict] = None,
                 cav_pts_neck: Optional[dict] = None,
                 drone_img_backbone: Optional[dict] = None,
                 drone_img_neck: Optional[dict] = None,
                 bbox_head: Optional[dict] = None,
                 cav_img_roi_head: Optional[dict] = None,
                 cav_img_rpn_head: Optional[dict] = None,
                 aggregator: Optional[dict] = None,
                 max_cav: int = 3,
                 max_drone: int = 2,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 **kwargs):
        super(VVMVXTwoStageDetector, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor, **kwargs)

        if cav_pts_voxel_encoder:
            self.cav_pts_voxel_encoder = MODELS.build(cav_pts_voxel_encoder)
        if cav_pts_middle_encoder:
            self.cav_pts_middle_encoder = MODELS.build(cav_pts_middle_encoder)
        if cav_pts_backbone:
            self.cav_pts_backbone = MODELS.build(cav_pts_backbone)
        if cav_pts_fusion_layer:
            self.cav_pts_fusion_layer = MODELS.build(cav_pts_fusion_layer)
        if cav_pts_neck is not None:
            self.cav_pts_neck = MODELS.build(cav_pts_neck)
        if bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            bbox_head.update(test_cfg=pts_test_cfg)
            self.bbox_head = MODELS.build(bbox_head)

        if cav_img_backbone:
            self.cav_img_backbone = MODELS.build(cav_img_backbone)
        if cav_img_neck is not None:
            self.cav_img_neck = MODELS.build(cav_img_neck)
        if cav_img_rpn_head is not None:
            self.cav_img_rpn_head = MODELS.build(cav_img_rpn_head)
        if cav_img_roi_head is not None:
            self.cav_img_roi_head = MODELS.build(cav_img_roi_head)
        if aggregator is not None:
            self.aggregator = MODELS.build(aggregator)
        
        if drone_img_backbone:
            self.drone_img_backbone = MODELS.build(drone_img_backbone)
        if drone_img_neck is not None:
            self.drone_img_neck = MODELS.build(drone_img_neck)
        self.max_cav = max_cav
        self.max_drone = max_drone
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_img_shared_head(self):
        """bool: Whether the detector has a shared head in image branch."""
        return hasattr(self,
                       'img_shared_head') and self.img_shared_head is not None

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'bbox_head') and self.bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self,
                       'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_cav_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'cav_img_backbone') and self.cav_img_backbone is not None

    @property
    def with_cav_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'cav_pts_backbone') and self.cav_pts_backbone is not None

    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer."""
        return hasattr(self,
                       'cav_pts_fusion_layer') and self.fusion_layer is not None

    @property
    def with_cav_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'cav_img_neck') and self.cav_img_neck is not None

    @property
    def with_cav_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'cav_pts_neck') and self.cav_pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'cav_img_rpn_head') and self.cav_img_rpn_head is not None

    @property
    def with_cav_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'cav_img_roi_head') and self.cav_img_roi_head is not None

    @property
    def with_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(self,
                       'voxel_encoder') and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(self,
                       'middle_encoder') and self.middle_encoder is not None

    @property
    def with_aggregator(self):
        """bool: Whether the detector has an aggregation head."""
        return hasattr(self,
                       'aggregator') and self.aggregator is not None
    
    @property
    def with_drone_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'drone_img_backbone') and self.drone_img_backbone is not None
    
    @property
    def with_drone_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'drone_img_neck') and self.drone_img_neck is not None

    def _forward(self):
        pass

    def extract_cav_img_feat(self, img: Tensor, input_metas: List[dict]) -> dict:
        """Extract features of images."""
        if self.with_cav_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in input_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.cav_img_backbone(img)
        else:
            return None
        if self.with_cav_img_neck:
            img_feats = self.cav_img_neck(img_feats)
        return img_feats


    def extract_drone_img_feat(self, img: Tensor, input_metas: List[dict]) -> dict:
        """Extract features of images."""
        if self.with_drone_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in input_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.drone_img_backbone(img)
        else:
            return None
        if self.with_drone_img_neck:
            img_feats = self.drone_img_neck(img_feats)
        return img_feats

    def extract_cav_pts_feat(
            self,
            voxel_dict: Dict[str, Tensor],
            points: Optional[List[Tensor]] = None,
            img_feats: Optional[Sequence[Tensor]] = None,
            batch_input_metas: Optional[List[dict]] = None
    ) -> Sequence[Tensor]:
        """Extract features of points.

        Args:
            voxel_dict(Dict[str, Tensor]): Dict of voxelization infos.
            points (List[tensor], optional):  Point cloud of multiple inputs.
            img_feats (list[Tensor], tuple[tensor], optional): Features from
                image backbone.
            batch_input_metas (list[dict], optional): The meta information
                of multiple samples. Defaults to True.

        Returns:
            Sequence[tensor]: points features of multiple inputs
            from backbone or neck.
        """
        if not self.with_bbox_head or not voxel_dict:
            return None
        voxel_features = self.cav_pts_voxel_encoder(voxel_dict['voxels'],
                                                voxel_dict['num_points'],
                                                voxel_dict['coors'], img_feats,
                                                batch_input_metas)
        batch_size = voxel_dict['coors'][-1, 0] + 1
        x = self.cav_pts_middle_encoder(voxel_features, voxel_dict['coors'],
                                    batch_size)
        x = self.cav_pts_backbone(x)
        if self.with_cav_pts_neck:
            x = self.cav_pts_neck(x)
        return x

    # dummy function for drone pts branch
    def extract_drone_pts_feat(
            self,
            voxel_dict: Dict[str, Tensor],
            points: Optional[List[Tensor]] = None,
            img_feats: Optional[Sequence[Tensor]] = None,
            batch_input_metas: Optional[List[dict]] = None
    ) -> Sequence[Tensor]:
        return None

    def remove_useless_agent(self, batch_inputs_dict: dict, agent_list: List[str], agent_num_gap: int, ego_vehicle: str):
        agent_idx = 0
        while agent_num_gap > 0:
            if agent_list[agent_idx] == ego_vehicle:
                agent_idx += 1
                continue
            else:
                cav_name = agent_list[agent_idx]
                batch_inputs_dict['infos'].pop(cav_name)
                agent_num_gap -= 1

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_input_metas: List[dict]) -> tuple:
        """Extract features from images and points.

        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains

                - points (List[tensor]):  Point cloud of multiple inputs.
                - imgs (tensor): Image tensor with shape (B, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
             tuple: Two elements in tuple arrange as
             image features and point cloud features.
        """
        agents = batch_inputs_dict['infos'].keys()
        ego_vehicle = batch_input_metas[0]['ego_vehicle']
        cav_list = sorted([agent for agent in agents if 'cav' in agent.lower()])
        cav_num = len(cav_list)
        cav_num_gap = cav_num - self.max_cav
        self.remove_useless_agent(batch_inputs_dict, cav_list, cav_num_gap, ego_vehicle)
        
        drone_list = sorted([agent for agent in agents if 'drone' in agent.lower()])
        drone_num = len(drone_list)
        drone_num_gap = drone_num - self.max_drone
        self.remove_useless_agent(batch_inputs_dict, drone_list, drone_num_gap, ego_vehicle)
        
        agents = batch_inputs_dict['infos'].keys()
        img_feats = dict()
        pts_feats = dict()
        for agent in agents:
            if 'cav' in agent.lower():
                extract_img_feat_func = self.extract_cav_img_feat
                extract_pts_feat_func = self.extract_cav_pts_feat
            elif 'drone' in agent.lower():
                extract_img_feat_func = self.extract_drone_img_feat
                extract_pts_feat_func = self.extract_drone_pts_feat
            else:
                raise NotImplementedError(f"Unknown agent {agent}!")
            voxel_dict = batch_inputs_dict['infos'][agent].get('voxels', None)
            imgs = batch_inputs_dict['infos'][agent].get('imgs', None)
            points = batch_inputs_dict['infos'][agent].get('points', None)
            img_feats[agent] = extract_img_feat_func(imgs, batch_input_metas)
            pts_feats[agent] = extract_pts_feat_func(
                voxel_dict,
                points=points,
                img_feats=img_feats,
                batch_input_metas=batch_input_metas)
        return (img_feats, pts_feats)

    def loss(self, batch_inputs_dict: Dict[List, torch.Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and `imgs` keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Tensor of batch images, has shape
                  (B, C, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        img_feats = {key: value for key, value in img_feats.items() if value is not None}
        pts_feats = {key: value for key, value in pts_feats.items() if value is not None}
        losses = dict()
        if pts_feats:
            if self.with_aggregator:
                agg_pts_feats = self.aggregator(pts_feats)
            else:
                pts_feats_list = list(pts_feats.values())
                agg_pts_feats = [sum(x) for x in zip(*pts_feats_list)]
            losses_pts = self.bbox_head.loss(agg_pts_feats, batch_data_samples,
                                                 **kwargs)
            losses.update(losses_pts)
        if img_feats:
            if self.with_aggregator:
                agg_img_feats = self.aggregator(img_feats)
            else:
                img_feats_list = list(img_feats.values())
                agg_img_feats = [sum(x) for x in zip(*img_feats_list)]
            losses_img = self.loss_imgs(agg_img_feats, batch_data_samples)
            losses.update(losses_img)
        return losses

    def loss_imgs(self, x: List[Tensor],
                  batch_data_samples: List[Det3DDataSample], **kwargs):
        """Forward function for image branch.

        This function works similar to the forward function of Faster R-CNN.

        Args:
            x (list[torch.Tensor]): Image features of shape (B, C, H, W)
                of multiple levels.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict: Losses of each branch.
        """
        losses = dict()
        # RPN forward and loss
        if self.with_img_rpn:
            proposal_cfg = self.test_cfg.rpn
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)
            rpn_losses, rpn_results_list = self.cav_img_rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg, **kwargs)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in keys:
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)

        else:
            if 'proposals' in batch_data_samples[0]:
                # use pre-defined proposals in InstanceData
                # for the second stage
                # to extract ROI features.
                rpn_results_list = [
                    data_sample.proposals for data_sample in batch_data_samples
                ]
            else:
                rpn_results_list = None
        # bbox head forward and loss
        if self.with_img_bbox:
            roi_losses = self.cav_img_roi_head.loss(x, rpn_results_list,
                                                batch_data_samples, **kwargs)
            losses.update(roi_losses)
        return losses

    def predict_imgs(self,
                     x: List[Tensor],
                     batch_data_samples: List[Det3DDataSample],
                     rescale: bool = True,
                     **kwargs) -> InstanceData:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            x (List[Tensor]): Image features from FPN.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.
        """

        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.cav_img_rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        results_list = self.cav_img_roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale, **kwargs)
        return results_list

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        img_feats = {key: value for key, value in img_feats.items() if value is not None}
        pts_feats = {key: value for key, value in pts_feats.items() if value is not None}
        if pts_feats and self.with_bbox_head:
            if self.with_aggregator:
                agg_pts_feats = self.aggregator(pts_feats)
            else:
                pts_feats_list = list(pts_feats.values())
                agg_pts_feats = [sum(x) for x in zip(*pts_feats_list)]
            results_list_3d = self.bbox_head.predict(
                agg_pts_feats, batch_data_samples, **kwargs)
        else:
            results_list_3d = None

        if img_feats and self.with_img_bbox:
            # TODO check this for camera modality
            if self.with_aggregator:
                agg_img_feats = self.aggregator(img_feats)
            else:
                img_feats_list = list(img_feats.values())
                agg_img_feats = [sum(x) for x in zip(*img_feats_list)]
            results_list_2d = self.predict_imgs(agg_img_feats, batch_data_samples,
                                                **kwargs)
        else:
            results_list_2d = None

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 results_list_3d,
                                                 results_list_2d)
        return detsamples