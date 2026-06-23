# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmengine.model import BaseModule
from torch import Tensor, nn as nn
from typing import List
from vvdet3d.registry import MODELS
from vvdet3d.utils import ConfigType, OptMultiConfig
import torch.nn.functional as F



def get_reference_points(H, W, bs=1, device='cuda', dtype=torch.float):
    """Get the reference points used in SCA and TSA.
    Args:
        H, W: spatial shape of bev.
        Z: hight of pillar.
        D: sample D points uniformly from each pillar.
        device (obj:`device`): The device where
            reference_points should be.
    Returns:
        Tensor: reference points used in decoder, has \
            shape (bs, num_keys, num_levels, 2).
    """
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            0.5, H - 0.5, H, dtype=dtype, device=device),
        torch.linspace(
            0.5, W - 0.5, W, dtype=dtype, device=device)
    )
    ref_y = ref_y.reshape(-1)[None] / H
    ref_x = ref_x.reshape(-1)[None] / W
    ref_2d = torch.stack((ref_x, ref_y), -1)
    ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
    return ref_2d



@MODELS.register_module()
class SpatialAttention(BaseModule):
    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 3,
                 num_points: int = 4,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 batch_first: bool = False,
                 ms_deform_attn_cfg: ConfigType = None,
                 ffn_cfg: ConfigType = None,
                 norm_cfg: ConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(SpatialAttention, self).__init__(init_cfg=init_cfg)
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # deformable attention
            if ms_deform_attn_cfg:
                deform_attn_cfg = ms_deform_attn_cfg.copy()
                deform_attn_cfg.update({
                    'embed_dims': embed_dims,
                    'num_heads': num_heads,
                    'num_levels': num_levels,
                    'num_points': num_points,
                    'dropout': dropout,
                    'batch_first': batch_first
                })
                deform_attn = MODELS.build(deform_attn_cfg)
            else:
                # default settings
                default_cfg = dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_levels=num_levels,
                    num_points=num_points,
                    dropout=dropout,
                    batch_first=batch_first
                )
                deform_attn = MODELS.build(default_cfg)
            
            # FFN
            if ffn_cfg:
                ffn = MODELS.build(ffn_cfg)
            else:
                # default settings
                default_ffn_cfg = dict(
                    type='FFN',
                    embed_dims=embed_dims,
                    feedforward_channels=embed_dims * 4,
                    num_fcs=2,
                    ffn_drop=dropout,
                    act_cfg=dict(type='ReLU', inplace=True)
                )
                ffn = MODELS.build(default_ffn_cfg)
            
            # LayerNorm
            norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
            norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
            
            layer = nn.ModuleDict({
                'self_attn': deform_attn,
                'norm1': norm1,
                'ffn': ffn,
                'norm2': norm2
            })
            self.layers.append(layer)



    def forward(self, 
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                identity: Tensor = None,
                query_pos: Tensor = None,
                key_padding_mask: Tensor = None,
                reference_points: Tensor = None,
                spatial_shapes: Tensor = None,
                level_start_index: Tensor = None,
                **kwargs) -> Tensor:
        """
        Multi-layer Deformable Attention forward pass
        """
        
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        # 逐层进行Deformable Attention
        for layer in self.layers:
            # Self-attention with residual connection and layer norm
            residual = query
            query = layer['norm1'](query)

            query = layer['self_attn'](
                query=query,
                key=key,
                value=value,
                identity=residual,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                **kwargs
            )

            # FFN with residual connection and layer norm
            residual = query
            query = layer['norm2'](query)
            query = layer['ffn'](query, identity=residual)

        return query





@MODELS.register_module()
class VVFormerAggregator(BaseModule):
    def __init__(self,
                 input_h: List[int] = None,
                 input_w: List[int] = None,
                 bev_h: int = None,
                 bev_w: int = None,
                 embed_dims: int = None,
                 agent_attention_cfg: ConfigType = None,
                 spatial_attention_cfg: ConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(VVFormerAggregator, self).__init__(init_cfg=init_cfg)
        self.input_h = input_h
        self.input_w = input_w
        self.embed_dims = embed_dims
        if spatial_attention_cfg:
            self.spatial_attention = MODELS.build(spatial_attention_cfg)
        if agent_attention_cfg:
            self.agent_attention = MODELS.build(agent_attention_cfg)
        self.bev_w = bev_w
        self.bev_h = bev_h

        
        # Agent-aware scoring module for each BEV position
        self.agent_scoring_net = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, 1),
            nn.Sigmoid()
        )

    def agent_fusion(self,
                     agent_features: List[List[Tensor]],
                     device: torch.device,
                     spatial_shapes_tensor: torch.Tensor,
                     reference_points: torch.Tensor) -> Tensor:
        """
        Cross-agent fusion:
        - Use ego (first CAV) BEV as query
        - Use other agents' BEV as key/value
        - Run deformable cross attention via `self.agent_attention` to obtain a per-position aggregated feature
        - Compute an agent score from the cross-attention output and fuse with ego feature

        Args:
            agent_features: list of per-agent multi-scale features, each [B, C, Hi, Wi]
            batch_size: B
            device: torch device
            spatial_shapes_tensor: [num_levels, 2] for one agent (per scale H, W)
            reference_points: [B, sum(H*W), num_levels, 2] for ego query
        Returns:
            aggregated_features: [B, sum(H*W), C]
        """
        num_levels = spatial_shapes_tensor.shape[0]

        # Build ego BEV sequence [B, sum(H*W), C]
        ego_feats = [f.flatten(2).transpose(1, 2) for f in agent_features[0]]
        ego_bev = torch.cat(ego_feats, dim=1)

        # If there are no other agents, return ego as-is
        if len(agent_features) == 1 or self.agent_attention is None:
            return ego_bev, ego_bev

        # Per-agent cross attention to compute scores, then select max-scored agent per position
        other_bev_list = []     # list of [B, S, C]
        score_list = []         # list of [B, S]
        for cav in agent_features[1:]:
            other_bev = torch.cat([f.flatten(2).transpose(1, 2) for f in cav], dim=1)  # [B, sum(H*W), C]
            other_bev_list.append(other_bev)

            # Use single-agent shapes and ego reference points
            agent_spatial_shapes = spatial_shapes_tensor  # [num_levels, 2]
            agent_level_start_index = torch.cat([
                agent_spatial_shapes.new_zeros((1,)),
                agent_spatial_shapes.prod(1).cumsum(0)[:-1]
            ])
            ref_pts = reference_points  # [B, sum(H*W), num_levels, 2]

            # Switch to (num_query, B, C) / (num_key, B, C)
            query = ego_bev.transpose(0, 1)   # [sum(H*W), B, C]
            key = other_bev.transpose(0, 1)   # [sum(H*W), B, C]
            value = key
            ref_pts_t = ref_pts.transpose(0, 1)  # [sum(H*W), B, num_levels, 2]

            cross_attn_out = self.agent_attention(
                query=query,
                key=key,
                value=value,
                reference_points=ref_pts_t,
                spatial_shapes=agent_spatial_shapes,
                level_start_index=agent_level_start_index
            )  # [sum(H*W), B, C]
            cross_attn_out = cross_attn_out.transpose(0, 1)  # [B, sum(H*W), C]

            # Score per BEV position from cross attention output
            score = self.agent_scoring_net(cross_attn_out).squeeze(-1)  # [B, sum(H*W)]
            score_list.append(score)

        # Stack and select best agent per position
        # shapes: others N, batch B, spatial S, channel C
        stacked_feats = torch.stack(other_bev_list, dim=0)  # [N, B, S, C]
        stacked_scores = torch.stack(score_list, dim=0)     # [N, B, S]
        best_idx = stacked_scores.argmax(dim=0)             # [B, S]

        # Gather features of the best agent: use gather on dim=2 after permuting to [B, S, N, C]
        feats_bsnc = stacked_feats.permute(1, 2, 0, 3)      # [B, S, N, C]
        gather_idx = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, feats_bsnc.size(-1))  # [B, S, 1, C]
        best_other = torch.gather(feats_bsnc, dim=2, index=gather_idx).squeeze(2)  # [B, S, C]
        # TODO: cross attention
        # aggregated_features = ego_bev + best_other
        return ego_bev, best_other

    def forward(self, agent_features: List[List[Tensor]]) -> List[torch.Tensor]:
        """
        Args:
            agent_features: List of length num_cav, each element is a list of multi-scale features [scale0, scale1, ...],
                         where each feature is [B, C, H, W] (or [C, H, W] if batch dim is merged).
        Returns:
            List[Tensor]: Aggregated features for each CAV (vehicle).
        """

        device = agent_features[0][0].device
        batch_size = agent_features[0][0].shape[0]
        # Use self.embed_dims for query and check consistency


        # Prepare spatial_shapes and level_start_index
        spatial_shapes = []
        for h, w in zip(self.input_h, self.input_w):
            spatial_shapes.append([h, w])
        spatial_shapes_tensor = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)  # [num_levels, 2]
        level_start_index = torch.cat([
            spatial_shapes_tensor.new_zeros((1,)),
            spatial_shapes_tensor.prod(1).cumsum(0)[:-1]
        ])  # [num_levels]



        # Prepare reference_points for deformable attention
        reference_points = get_reference_points(self.bev_h, self.bev_w, bs=batch_size, device=device)  # [B, sum(H*W), 2]
        num_levels = len(self.input_h)
        reference_points = reference_points.repeat(1, 1, num_levels, 1)  # [B, sum(H*W), num_levels, 2]
        
        # Cross-agent fusion using deformable cross attention to compute scores
        aggregated_query, aggregated_key_value = self.agent_fusion(
            agent_features,
            device,
            spatial_shapes_tensor,
            reference_points,
        )
        aggregated_query = aggregated_query.transpose(0, 1)  # [sum(H*W), B, embed_dims]
        aggregated_key_value = aggregated_key_value.transpose(0, 1)  # [sum(H*W), B, embed_dims]
        reference_points = reference_points.transpose(0, 1)  # [sum(H*W), B, num_levels, 2]
        
        # Use aggregated features as key and value
        out = self.spatial_attention(
            query=aggregated_query,
            key=aggregated_key_value,
            value=aggregated_key_value,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes_tensor,
            level_start_index=level_start_index
        )
        
        # Convert back to batch_first=True format: [B, sum(H*W), embed_dims]
        out = out.transpose(0, 1)  # [B, sum(H*W), embed_dims]
        
        # Split and reshape back to multi-scale format: List[[B, C, H, W]]
        output_list = []
        start_idx = 0
        for h, w in zip(self.input_h, self.input_w):
            end_idx = start_idx + h * w
            scale_out = out[:, start_idx:end_idx, :]  # [B, H*W, C]
            scale_out = scale_out.transpose(1, 2).reshape(batch_size, self.embed_dims, h, w)  # [B, C, H, W]
            output_list.append(scale_out)
            start_idx = end_idx
        
        # Return single output for all vehicles (aggregated result)
        return output_list
