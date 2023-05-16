import torch.nn as nn
from .modules import Flatten, Activation

import torch.nn.functional as F


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)


# projection head for contrast learning
class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim, num_proj_layer):
        super().__init__()
        if num_proj_layer == 1:
            self.net = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif num_proj_layer == 2: # TODO: support multi-layers maybe
            self.net = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1),
            )
        else:
            raise NotImplementedError("Only support Linear or 2-layer now")

    def forward(self, x):
        features = self.net(x)
        features = F.normalize(features, p=2, dim=1) # normalize projected feature
        return features
