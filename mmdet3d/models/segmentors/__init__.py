# Copyright (c) OpenMMLab. All rights reserved.
from .base import Base3DSegmentor
from .encoder_decoder import EncoderDecoder3D
from .modelnet40_encoder import ModelNetEncoderDecoder3D

__all__ = ['Base3DSegmentor', 'EncoderDecoder3D', 'ModelNetEncoderDecoder3D']
