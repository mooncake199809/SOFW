_base_ = [
    '../_base_/datasets/s3dis-3d-5class.py', '../_base_/models/votenet.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]

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
            mean_sizes= [[1.7744198,  1.4419658, 0.65777165],
                        [0.6254625,  0.63793474, 0.7833646],
                        [1.200909,  0.9999453, 0.7396],
                        [1.2822456, 0.666723, 1.4165214],
                        [1.3756422, 0.64827, 1.267781],]
        )))

log_config = dict(interval=10)