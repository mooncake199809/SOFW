_base_ = [
    '../_base_/datasets/s3dis-3d-5class.py', '../_base_/models/votenet.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]

custom_imports=dict(imports='mmcls.models', allow_failed_imports=False) 

# model settings
model = dict(
    backbone=dict(conv_cfg=dict(type='Conv2d_share')),
    bbox_head=dict(
        num_classes=5,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=5,
            num_dir_bins=1,
            with_rot=False,
            # mean_sizes=[[2.1171012, 1.7463478, 0.5963188],
            #             [0.6533775,  0.65529823, 0.8648145],
            #             [1.5875,    1.2321666, 0.7266666],
            #             [0.9222696, 0.5310676, 1.3631234],
            #             [1.7465187, 0.3827408, 1.360889 ]],
            mean_sizes= [[1.7744198,  1.4419658, 0.65777165],
                        [0.6254625,  0.63793474, 0.7833646],
                        [1.200909,  0.9999453, 0.7396],
                        [1.2822456, 0.666723, 1.4165214],
                        [1.3756422, 0.64827, 1.267781],]
        )))

# yapf:disable
log_config = dict(interval=10)
# yapf:enable

# load_from = '/home/xietao/xt_dataset/joint_3d_dk/mtl_2xchannel/_task_s3dis/iter_20000.pth'

# optimizer = dict(type='AdamW', lr=0.008, weight_decay=0.05)
# optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# lr_config = dict(
#     _delete_=True,
#     policy='CosineAnnealing',
#     min_lr=0,
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1e-4)
# runner = dict(_delete_=True, type="IterBasedRunner", max_iters=20000)
# evaluation = dict(
#     by_epoch = False,
#     interval = 1000
# )
# checkpoint_config = dict(interval=1000, max_keep_ckpts=5)
# find_unused_parameters=True
