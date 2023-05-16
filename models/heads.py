from torch import nn
import torch.nn.functional as F

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


class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, x):
        return x