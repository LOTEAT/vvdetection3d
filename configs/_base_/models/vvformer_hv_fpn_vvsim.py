# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
voxel_size = [0.5, 0.5, 8]
model = dict(
    type='VVFormer',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=64,
            point_cloud_range=[-50, -50, -5, 50, 50, 3],
            voxel_size=voxel_size,
            max_voxels=(30000, 40000))),
    cav_pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=3,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),
    cav_pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[200, 200]),
    cav_pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    cav_pts_neck=dict(
        type='mmdet.FPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=3),
    drone_img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[2,],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa: E501
        )
    ),
    drone_img_neck=dict(
        type="CrossViewSampler",
        cross_view_shape=[[100, 100],],
        point_cloud_range=[-50, -50, -1, 50, 50, 2],
        d_vox=4,
        feat_channels=[384,],
        embed_dim=384,
    ),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=2,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            scales=[1, 2, 4],
            sizes=[
                [2.5981, 0.8660, 1.],  # 1.5 / sqrt(3)
                [1.7321, 0.5774, 1.],  # 1 / sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    aggregator=dict(
        type="VVFormerAggregator",
        input_h=[100, ],
        input_w=[100, ],
        bev_h=30,
        bev_w=30,
        embed_dims=384,
        spatial_attention_cfg=dict(
            type='SpatialAttention',
            embed_dims=384,
            num_heads=8,
            num_levels=1,
            num_points=4,
            num_layers=1,
            dropout=0.1,
            batch_first=False,
            norm_cfg=dict(type='LN'),
            ms_deform_attn_cfg=dict(
                type='MultiScaleDeformableAttention',
                embed_dims=384,
                num_heads=8,
                num_levels=1,
                num_points=4,
                dropout=0.1,
                batch_first=False
            )
        ),
        agent_attention_cfg=dict(
            type='SpatialAttention',
            embed_dims=384,
            num_heads=8,
            num_levels=1,
            num_points=4,
            num_layers=1,
            dropout=0.1,
            batch_first=False,
            norm_cfg=dict(type='LN'),
            ms_deform_attn_cfg=dict(
                type='MultiScaleDeformableAttention',
                embed_dims=384,
                num_heads=8,
                num_levels=1,
                num_points=4,
                dropout=0.1,
                batch_first=False
            )
        )
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500)))
