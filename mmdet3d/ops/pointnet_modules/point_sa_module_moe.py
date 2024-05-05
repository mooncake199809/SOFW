# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmcv.cnn import ConvModule
from mmcv.ops import GroupAll
from mmcv.ops import PointsSampler as Points_Sampler
from mmcv.ops import QueryAndGroup, gather_points
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.ops import PAConv
from .builder import SA_MODULES


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class MOE_MLP(nn.Module):
    def __init__(self, expert_module, n_embed, out_channel):
        super().__init__()
        self.n_embed = n_embed
        self.topk_num = 2
        self.experts_num = 3
        self.out_channel = out_channel
        self.experts_list = nn.ModuleList([expert_module for i in range(self.experts_num)])

        self.experts_logit = nn.Linear(self.n_embed, self.experts_num)
        self.noise_logit = nn.Linear(self.n_embed, self.experts_num)
    
    def forward(self, x):
        B, C, N, P = x.shape # N: 球的个数 P: 每个球内点的个数
        A = B * N * P
        x = x.permute(0, 2, 3, 1).reshape(A, C)

        output_final = torch.zeros(A, self.out_channel).type_as(x)
        expert_logit_ = self.experts_logit(x)                               # [N Expert]
        noise_logit_ = self.noise_logit(x)                                  # [N Expert]

        noise = torch.rand_like(expert_logit_)
        noise = noise * F.softmax(noise_logit_, dim=-1)                     # [N Expert]
        expert_logit_ = expert_logit_ + noise                               # [N Expert]
        value, index = torch.topk(expert_logit_, dim=-1, k=self.topk_num)   # [N 2]
        expert_logit_final = torch.full_like(expert_logit_, float("-inf"))
        expert_logit_final[torch.arange(A)[:, None].repeat(1,self.topk_num), index] = value
        expert_logit_final = F.softmax(expert_logit_final, dim=-1)          # [N Expert]

        for i, expert_act in enumerate(self.experts_list):
            token_act_index = (index == i)                                  # [N 2]
            token_act_index_T = (index == i).any(dim=-1)                    # [N]

            if token_act_index_T.any():
                token_act = x[token_act_index_T, :]                         # [P C]
                print("--------------------------------------------------------------")
                print(i, token_act_index_T.shape, token_act.shape)
                token_act = expert_act(token_act[:, :, None, None])[:,:,0,0]
                logit_value = expert_logit_final[torch.where(token_act_index == True)][:, None] # [P 1]
                output_final[token_act_index_T, :] += logit_value * token_act
        
        return output_final.reshape(B, N, P, self.out_channel).permute(0, -1, 1, 2)

class BasePointSAModule_MOE(nn.Module):
    """Base module for point set abstraction module used in PointNets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str], optional): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int], optional):
            Range of points to apply FPS. Default: [-1].
        dilated_group (bool, optional): Whether to use dilated ball query.
            Default: False.
        use_xyz (bool, optional): Whether to use xyz.
            Default: True.
        pool_mod (str, optional): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool, optional): Whether to normalize local XYZ
            with radius. Default: False.
        grouper_return_grouped_xyz (bool, optional): Whether to return
            grouped xyz in `QueryAndGroup`. Defaults to False.
        grouper_return_grouped_idx (bool, optional): Whether to return
            grouped idx in `QueryAndGroup`. Defaults to False.
    """

    def __init__(self,
                 num_point,
                 radii,
                 sample_nums,
                 mlp_channels,
                 mlp_channels_inverted,
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 dilated_group=False,
                 use_xyz=True,
                 pool_mod='max',
                 normalize_xyz=False,
                 grouper_return_grouped_xyz=False,
                 grouper_return_grouped_idx=False):
        super(BasePointSAModule_MOE, self).__init__()

        assert len(radii) == len(sample_nums) == len(mlp_channels)
        assert pool_mod in ['max', 'avg']
        assert isinstance(fps_mod, list) or isinstance(fps_mod, tuple)
        assert isinstance(fps_sample_range_list, list) or isinstance(
            fps_sample_range_list, tuple)
        assert len(fps_mod) == len(fps_sample_range_list)

        if isinstance(mlp_channels, tuple):
            mlp_channels = list(map(list, mlp_channels))
        self.mlp_channels = mlp_channels
        self.mlp_channels_inverted = mlp_channels_inverted

        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        elif num_point is None:
            self.num_point = None
        else:
            raise NotImplementedError('Error type of num_point!')

        self.pool_mod = pool_mod
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.inverted_mlps = nn.ModuleList()
        self.relu_list = nn.ModuleList()
        self.inverted_relu_list = nn.ModuleList()
        self.shortcut = nn.ModuleList()
        self.shortcut_inverted = nn.ModuleList()

        self.fps_mod_list = fps_mod
        self.fps_sample_range_list = fps_sample_range_list

        if self.num_point is not None:
            self.points_sampler = Points_Sampler(self.num_point,
                                                 self.fps_mod_list,
                                                 self.fps_sample_range_list)
        else:
            self.points_sampler = None

        for i in range(len(radii)):
            radius = radii[i]
            sample_num = sample_nums[i]
            if num_point is not None:
                if dilated_group and i != 0:
                    min_radius = radii[i - 1]
                else:
                    min_radius = 0
                grouper = QueryAndGroup(
                    radius,
                    sample_num,
                    min_radius=min_radius,
                    use_xyz=use_xyz,
                    normalize_xyz=normalize_xyz,
                    return_grouped_xyz=grouper_return_grouped_xyz,
                    return_grouped_idx=grouper_return_grouped_idx)
            else:
                grouper = GroupAll(use_xyz)
            self.groupers.append(grouper)


    def _sample_points(self, points_xyz, features, indices, target_xyz):
        """Perform point sampling based on inputs.

        If `indices` is specified, directly sample corresponding points.
        Else if `target_xyz` is specified, use is as sampled points.
        Otherwise sample points using `self.points_sampler`.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
            indices (Tensor): (B, num_point) Index of the features.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tensor: (B, num_point, 3) sampled xyz coordinates of points.
            Tensor: (B, num_point) sampled points' index.
        """
        xyz_flipped = points_xyz.transpose(1, 2).contiguous()
        if indices is not None:
            assert (indices.shape[1] == self.num_point[0])
            new_xyz = gather_points(xyz_flipped, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None
        elif target_xyz is not None:
            new_xyz = target_xyz.contiguous()
        else:
            if self.num_point is not None:
                indices = self.points_sampler(points_xyz, features)
                new_xyz = gather_points(xyz_flipped,
                                        indices).transpose(1, 2).contiguous()
            else:
                new_xyz = None

        return new_xyz, indices

    def _pool_features(self, features):
        """Perform feature aggregation using pooling operation.

        Args:
            features (torch.Tensor): (B, C, N, K)
                Features of locally grouped points before pooling.

        Returns:
            torch.Tensor: (B, C, N)
                Pooled features aggregating local information.
        """
        if self.pool_mod == 'max':
            # (B, C, N, 1)
            new_features = F.max_pool2d(
                features, kernel_size=[1, features.size(3)])
        elif self.pool_mod == 'avg':
            # (B, C, N, 1)
            new_features = F.avg_pool2d(
                features, kernel_size=[1, features.size(3)])
        else:
            raise NotImplementedError

        return new_features.squeeze(-1).contiguous()

    def forward(
        self,
        points_xyz,
        features=None,
        index=None,             # Modified by DK        
        indices=None,
        target_xyz=None,
    ):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor, optional): (B, C, N) features of each point.
                Default: None.
            indices (Tensor, optional): (B, num_point) Index of the features.
                Default: None.
            target_xyz (Tensor, optional): (B, M, 3) new coords of the outputs.
                Default: None.

        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
                Index of the features.
        """
        new_features_list = []

        # sample points, (B, num_point, 3), (B, num_point)
        # num_point: 球的个数   nsample: 每个球内点的个数
        new_xyz, indices = self._sample_points(points_xyz, features, indices,
                                               target_xyz)
        
        for i in range(len(self.groupers)):
            # grouped_results may contain:
            # - grouped_features: (B, C, num_point, nsample)
            # - grouped_xyz: (B, 3, num_point, nsample)
            # - grouped_idx: (B, num_point, nsample)
            grouped_results = self.groupers[i](points_xyz, new_xyz, features)

            # (B, mlp[-1], num_point, nsample)
            new_features = self.mlps[i](grouped_results)
            if self.shortcut[i] is not None:
                new_features = self.relu_list[i](new_features + 
                                                 self.shortcut[i](grouped_results))
            else:
                new_features = self.relu_list[i](new_features + grouped_results)

            new_features_inverted = self.inverted_mlps[i](new_features)
            if self.shortcut_inverted[i] is not None:
                new_features = self.inverted_relu_list[i](new_features_inverted + 
                                                          self.shortcut_inverted[i](new_features))
            else:
                new_features = self.inverted_relu_list[i](new_features_inverted + new_features)

            # this is a bit hack because PAConv outputs two values
            # we take the first one as feature
            if isinstance(self.mlps[i][0], PAConv):
                assert isinstance(new_features, tuple)
                new_features = new_features[0]

            # (B, mlp[-1], num_point)
            new_features = self._pool_features(new_features)

            # new_features = self.inverted_mlps[i](new_features[:,:,None,:])[:,:,0,:]
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1), indices


@SA_MODULES.register_module()
class PointSAModuleMSG_MOE(BasePointSAModule_MOE):
    """Point set abstraction module with multi-scale grouping (MSG) used in
    PointNets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[list[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str], optional): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int], optional): Range of points to
            apply FPS. Default: [-1].
        dilated_group (bool, optional): Whether to use dilated ball query.
            Default: False.
        norm_cfg (dict, optional): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool, optional): Whether to use xyz.
            Default: True.
        pool_mod (str, optional): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool, optional): Whether to normalize local XYZ
            with radius. Default: False.
        bias (bool | str, optional): If specified as `auto`, it will be
            decided by `norm_cfg`. `bias` will be set as True if
            `norm_cfg` is None, otherwise False. Default: 'auto'.
    """

    def __init__(self,
                 num_point,
                 radii,
                 sample_nums,
                 mlp_channels,
                 mlp_channels_inverted,
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 dilated_group=False,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=True,
                 pool_mod='max',
                 normalize_xyz=False,
                 bias='auto'):
        super(PointSAModuleMSG_MOE, self).__init__(
            num_point=num_point,
            radii=radii,
            sample_nums=sample_nums,
            mlp_channels=mlp_channels,
            mlp_channels_inverted = mlp_channels_inverted,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            dilated_group=dilated_group,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            normalize_xyz=normalize_xyz)

        for i in range(len(self.mlp_channels)):
            mlp_channel = self.mlp_channels[i]
            mlp_channel_inverted = mlp_channels_inverted[i]
            if use_xyz:
                mlp_channel[0] += 3

            mlp = nn.Sequential()
            mlp_inverted = nn.Sequential()
            for i in range(len(mlp_channel) - 1):
                if i == 0:
                    first_in_channel = mlp_channel[i]
                    second_in_channel = mlp_channel_inverted[i]
                if i == len(mlp_channel) - 2:
                    first_out_channel = mlp_channel[i + 1]
                    second_out_channel = mlp_channel_inverted[i + 1]
                
                if i == len(mlp_channel) - 2:
                    act_cfg_value = None
                else:
                    act_cfg_value = dict(type="ReLU")    
                
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        mlp_channel[i], mlp_channel[i + 1], kernel_size=(1, 1), stride=(1, 1),
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg = act_cfg_value, bias=bias))
                mlp_inverted.add_module(
                    f'inverted_layer{i}',
                    ConvModule(
                        mlp_channel_inverted[i], mlp_channel_inverted[i + 1], kernel_size=(1, 1), stride=(1, 1),
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg = act_cfg_value, bias=bias))
                    

                # mlp.add_module(
                #     f'layer{i}',
                #     MOE_MLP(expert_module=ConvModule(
                #         mlp_channel[i], mlp_channel[i + 1], kernel_size=(1, 1), stride=(1, 1),
                #         conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg = act_cfg_value, bias=bias), 
                #         n_embed=mlp_channel[i], out_channel=mlp_channel[i + 1]))
                # mlp_inverted.add_module(
                #     f'inverted_layer{i}',
                #     MOE_MLP(expert_module=ConvModule(
                #         mlp_channel_inverted[i], mlp_channel_inverted[i + 1], kernel_size=(1, 1), stride=(1, 1),
                #         conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg = act_cfg_value, bias=bias), 
                #         n_embed=mlp_channel_inverted[i], out_channel=mlp_channel_inverted[i + 1]))

            if first_in_channel == first_out_channel:  
                    self.shortcut.append(None)
            else:
                self.shortcut.append(
                    ConvModule(
                    first_in_channel,
                    first_out_channel,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg = None,
                    bias=bias)
                )
            
            if second_in_channel == second_out_channel:
                self.shortcut_inverted.append(None)
            else:
                self.shortcut_inverted.append(
                    ConvModule(
                    second_in_channel,
                    second_out_channel,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg = None,
                    bias=bias)
                )
            
            self.relu_list.append(nn.ReLU())
            self.inverted_relu_list.append(nn.ReLU())
            self.mlps.append(mlp)
            self.inverted_mlps.append(mlp_inverted)
        
        # self.global_att = TransformerBlock(mlp_channel[-1], 256, 64)
        


@SA_MODULES.register_module()
class PointSAModule_MOE(PointSAModuleMSG_MOE):
    """Point set abstraction module with single-scale grouping (SSG) used in
    PointNets.

    Args:
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        num_point (int, optional): Number of points.
            Default: None.
        radius (float, optional): Radius to group with.
            Default: None.
        num_sample (int, optional): Number of samples in each ball query.
            Default: None.
        norm_cfg (dict, optional): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool, optional): Whether to use xyz.
            Default: True.
        pool_mod (str, optional): Type of pooling method.
            Default: 'max_pool'.
        fps_mod (list[str], optional): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
        fps_sample_range_list (list[int], optional): Range of points
            to apply FPS. Default: [-1].
        normalize_xyz (bool, optional): Whether to normalize local XYZ
            with radius. Default: False.
    """

    def __init__(self,
                 mlp_channels,
                 mlp_channels_inverted,
                 num_point=None,
                 radius=None,
                 num_sample=None,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 use_xyz=True,
                 pool_mod='max',
                 fps_mod=['D-FPS'],
                 fps_sample_range_list=[-1],
                 normalize_xyz=False):
        super(PointSAModule_MOE, self).__init__(
            mlp_channels=[mlp_channels],
            mlp_channels_inverted=[mlp_channels_inverted],
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            normalize_xyz=normalize_xyz)
