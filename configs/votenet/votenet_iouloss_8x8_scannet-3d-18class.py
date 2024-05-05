_base_ = ['./votenet_8x8_scannet-3d-18class.py']

custom_imports=dict(imports='mmcls.models', allow_failed_imports=False) 

# model settings, add iou loss
model = dict(
    backbone=dict(conv_cfg=dict(type='Conv2d_share')),
    bbox_head=dict(
        iou_loss=dict(
            type='AxisAlignedIoULoss', reduction='sum', loss_weight=10.0 /
            3.0)))

load_from = '/home/xietao/xt_dataset/joint_3d_dk/mtl_2xchannel/_task_scannet/iter_20000.pth'
find_unused_parameters=True
