import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmengine.model import BaseModule
from torch import nn as nn
from typing import List, Tuple
from vvdet3d.registry import MODELS
from vvdet3d.utils import ConfigType, OptMultiConfig
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

@MODELS.register_module()
class CrossViewSampler(BaseModule):
    def __init__(self,
                 cross_view_shape: List[Tuple] = None,
                 point_cloud_range: list = None,
                 d_vox: int = None,
                 feat_channels: List[int] = None,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 init_cfg: OptMultiConfig = None):
        super(CrossViewSampler, self).__init__(init_cfg=init_cfg)
        self.point_cloud_range = point_cloud_range
        self.pc_min = torch.tensor(point_cloud_range[:3], dtype=torch.float32)
        self.pc_max = torch.tensor(point_cloud_range[3:], dtype=torch.float32)
        self.cross_view_shape = cross_view_shape
        self.d_vox = d_vox
        self.embed_dim = embed_dim
        
        # Height-aware Transformer
        self.height_fusion = HeightAwareTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            d_vox=d_vox
        )
        
        # Feature adapters - will be built dynamically
        self.feat_adapters = nn.ModuleList()
        self._build_adapters(feat_channels)

    def _build_adapters(self, feat_channels: List[int]):
        """动态构建特征适配器"""
        if len(self.feat_adapters) == 0:
            for ch in feat_channels:
                self.feat_adapters.append(
                    nn.Conv2d(ch, self.embed_dim, 1)
                )

    def forward(self, feat: List[torch.Tensor],
                proj_mats: List[torch.Tensor], 
                downsample_rates: List[int]) -> List[torch.Tensor]:
        
        # 动态构建适配器
        
        # 只处理第一个尺度，简化流程
        f = feat[0]
        B, C, H_img, W_img = f.shape
        H_vox, W_vox = self.cross_view_shape[0]
        D_vox = self.d_vox
        proj_mat = proj_mats[0]
        ds_rate = downsample_rates[0]
        device = f.device

        # 1. 生成 voxel 网格坐标
        xs = torch.linspace(self.pc_min[0], self.pc_max[0], W_vox, device=device)
        ys = torch.linspace(self.pc_min[1], self.pc_max[1], H_vox, device=device)
        zs = torch.linspace(self.pc_min[2], self.pc_max[2], D_vox, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing='ij')
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).permute(1,2,0,3)
        grid = grid.unsqueeze(0).repeat(B,1,1,1,1)

        # 2. 投影和有效性检查
        ones = torch.ones(B, H_vox, W_vox, D_vox, 1, device=device)
        grid_homo = torch.cat([grid, ones], dim=-1)
        grid_flat = grid_homo.view(B, -1, 4).transpose(1,2)
        uvw = torch.bmm(proj_mat, grid_flat)
        
        # 深度和边界检查
        depth_mask = uvw[:,2,:] > 0.1
        u = uvw[:,0,:] / (uvw[:,2,:]+1e-6)
        v = uvw[:,1,:] / (uvw[:,2,:]+1e-6)
        u = u / ds_rate
        v = v / ds_rate
        
        valid_u = (u >= 0) & (u < W_img)
        valid_v = (v >= 0) & (v < H_img)
        valid_mask = valid_u & valid_v & depth_mask
        valid_mask = valid_mask.view(B, H_vox, W_vox, D_vox)

        # 3. 特征采样
        u_norm = 2 * u / (W_img - 1) - 1
        v_norm = 2 * v / (H_img - 1) - 1
        grid_sample_2d = torch.stack([u_norm, v_norm], dim=-1)
        grid_sample_2d = grid_sample_2d.view(B, H_vox, W_vox, D_vox, 2)

        f_adapted = self.feat_adapters[0](f)  # (B, embed_dim, H_img, W_img)
        
        f_expand = f_adapted.unsqueeze(1).expand(B, D_vox, self.embed_dim, H_img, W_img)
        f_expand = f_expand.reshape(B*D_vox, self.embed_dim, H_img, W_img)
        grid_expand = grid_sample_2d.permute(0,3,1,2,4).reshape(B*D_vox, H_vox, W_vox, 2)
        
        sampled = F.grid_sample(
            f_expand, grid_expand,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        sampled = sampled.view(B, D_vox, self.embed_dim, H_vox, W_vox)

        # 4. Height-aware融合
        fused_features = self.height_fusion(sampled, valid_mask)
        
        return [fused_features]




class HeightAwareTransformer(nn.Module):
    """向量化的高度感知Transformer"""
    
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, 
                 dropout: float, d_vox: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_vox = d_vox
        
        # 恢复之前的高度位置编码
        self.height_pos_encoding = HeightPositionalEncoding(embed_dim, d_vox)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                activation='relu',
                batch_first=True,
                norm_first=True  # 使用pre-norm，更稳定
            ) for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, sampled_feat: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sampled_feat: (B, D_vox, embed_dim, H_vox, W_vox)
            valid_mask: (B, H_vox, W_vox, D_vox)
        Returns:
            fused_feat: (B, embed_dim, H_vox, W_vox)
        """
        B, D_vox, C, H_vox, W_vox = sampled_feat.shape
        device = sampled_feat.device
        
        # 重排为 (B, H_vox, W_vox, D_vox, embed_dim)
        feat_spatial = sampled_feat.permute(0, 3, 4, 1, 2)  # (B, H_vox, W_vox, D_vox, C)
        mask_spatial = valid_mask  # (B, H_vox, W_vox, D_vox)
        
        # 向量化处理：reshape为 (B*H_vox*W_vox, D_vox, embed_dim)
        batch_size_flat = B * H_vox * W_vox
        height_feats = feat_spatial.view(batch_size_flat, D_vox, C)  # (B*H*W, D_vox, C)
        height_masks = mask_spatial.view(batch_size_flat, D_vox)     # (B*H*W, D_vox)
        
        # 添加高度位置编码
        position_ids = torch.arange(D_vox, device=device)  # (D_vox,)
        height_pos = self.height_pos_encoding(position_ids)  # (D_vox, embed_dim)
        height_feats = height_feats + height_pos.unsqueeze(0)  # (B*H*W, D_vox, C)
        
        # 特征归一化（防止数值过大）
        height_feats = F.layer_norm(height_feats, height_feats.shape[-1:])
        

        
        # 创建注意力mask（True表示要忽略的位置）
        attn_mask = ~height_masks  # (B*H*W, D_vox)
        
        # 检查是否存在全部被mask的序列
        all_masked = attn_mask.all(dim=1)  # (B*H*W,)
        if all_masked.any():
            # 对于全部被mask的序列，至少保留一个位置
            for i in range(attn_mask.size(0)):
                if all_masked[i]:
                    attn_mask[i, 0] = False  # 保留第一个位置
        
        # 通过Transformer层（向量化处理）
        x = height_feats
        for layer_idx, layer in enumerate(self.transformer_layers):
            # 添加调试信息
            x = layer(x, src_key_padding_mask=attn_mask)

        x = self.output_norm(x)  # (B*H*W, D_vox, embed_dim)
        
        # 加权平均融合（向量化）
        weights = height_masks.float().unsqueeze(-1)  # (B*H*W, D_vox, 1)
        weighted_feat = (x * weights).sum(dim=1)      # (B*H*W, embed_dim
        # 重排回 (B, H_vox, W_vox, embed_dim) 然后转为 (B, embed_dim, H_vox, W_vox)
        fused_output = weighted_feat.view(B, H_vox, W_vox, C)
        return fused_output.permute(0, 3, 1, 2)

class HeightPositionalEncoding(nn.Module):
    """高度维度的位置编码"""
    
    def __init__(self, embed_dim: int, max_height: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        pe = torch.zeros(max_height, embed_dim)
        position = torch.arange(0, max_height, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position_ids: (seq_len,) 位置索引
        Returns:
            pos_encoding: (seq_len, embed_dim)
        """
        return self.pe[position_ids]