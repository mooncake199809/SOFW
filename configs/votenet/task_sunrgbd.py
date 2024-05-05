_base_ = [
    './votenet_16x8_sunrgbd-3d-10class.py'
]

custom_imports=dict(imports='mmcls.models', allow_failed_imports=False) 

# data = dict(
#     train=dict(
#         _delete_ = True,
#         type={{_base_.dataset_type}},
#         data_root={{_base_.data_root}},
#         ann_file='/data/xt_dataset/sunrgbd/sunrgbd_processed/' + 'sunrgbd_infos_train.pkl',
#         pipeline={{_base_.train_pipeline}},
#         classes={{_base_.class_names}},
#         filter_empty_gt=False,
#         # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
#         # and box_type_3d='Depth' in sunrgbd and scannet dataset.
#         box_type_3d='Depth'))


model = dict(
    backbone=dict(conv_cfg=dict(type='Conv2d_share')),
    # bbox_head=dict(vote_module_cfg=dict(conv_cfg=dict(type='Conv1d_share')),
    #                vote_aggregation_cfg=dict(conv_cfg=dict(type='Conv2d_share')))
)

data = dict(samples_per_gpu=16)

max_iters = 20000

test_setting = dict(
    repo='mmdet3d',  # call which repo's test function
    single_gpu_test=dict(),
    multi_gpu_test=dict())

evaluation = dict(
    type='mme.MultiTaskEvalHook',
    dataset={{_base_.data.val}},
    dataloader=dict(samples_per_gpu=1, workers_per_gpu=12),
    test_setting=test_setting,
    by_epoch=False,
    interval=1000)

optimizer = dict(type='AdamW', lr=0.008, weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-4)
runner = dict(_delete_=True, type='mme.MultiTaskIterBasedRunner', max_iters=max_iters)

# runtime
checkpoint_config = dict(interval=1000, max_keep_ckpts=5)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='mmcv.TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])

# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', max_iters)]