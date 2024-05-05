# Copyright (c) OpenMMLab. All rights reserved.
from .dgcnn_head import DGCNNHead
from .paconv_head import PAConvHead
from .pointnet2_head import PointNet2Head
from .pointnet2_head_cls import PointNet2HeadCLS

__all__ = ['PointNet2Head', 'DGCNNHead', 'PAConvHead', 'PointNet2HeadCLS']
