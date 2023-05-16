import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md


# --------------- SRM ---------------
class SRMConv2D(nn.Module):
    def __init__(self, stride=1, padding=2):
        super(SRMConv2D, self).__init__()
        self.stride = stride
        self.padding = padding
        self.SRMWeights = nn.Parameter(
            self._get_srm_list(), requires_grad=False)

    def _get_srm_list(self):
        # srm kernel 1
        srm1 = [[0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0]]
        srm1 = torch.tensor(srm1, dtype=torch.float32) / 4.

        # srm kernel 2
        srm2 = [[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]]
        srm2 = torch.tensor(srm2, dtype=torch.float32) / 12.

        # srm kernel 3
        srm3 = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
        srm3 = torch.tensor(srm3, dtype=torch.float32) / 2.

        return torch.stack([torch.stack([srm1, srm1, srm1], dim=0), torch.stack([srm2, srm2, srm2], dim=0),
                            torch.stack([srm3, srm3, srm3], dim=0)], dim=0)

    def forward(self, X):
        return F.conv2d(X, self.SRMWeights, stride=self.stride, padding=self.padding)

# --------------- CONSTRAINED CONV ---------------
class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super(BayarConv2d, self).__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)

    def bayarConstraint(self):
        # length == kernel_size ** 2 - 1
        # in_channels * out_channels * length
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        # get the center position
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]),
                                dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x


# --------------- REGULAR CONV + SRM + CONTRAINED CONV ---------------
class CombinedConv2D(nn.Module):
    def __init__(self, in_channels, kernel_size=5, stride=1, padding=2):
        super(CombinedConv2D, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv =nn.Conv2d(in_channels=self.in_channels, out_channels=10, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False)
        self.srm = SRMConv2D(stride=self.stride)
        self.bayar = BayarConv2d(in_channels=self.in_channels, out_channels=3, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(3)

        self.attention = md.Attention('scse', in_channels=16)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.srm(x)
        x3 = self.bayar(x)

        x_cat = torch.cat([x1, x2, x3], dim=1)
        x_att = self.attention(x_cat)

        # x: torch.Size([1, 3, 512, 512])
        # conv(x): torch.Size([1, 10, 512, 512])
        # srm(x): torch.Size([1, 3, 512, 512])
        # bayar(x): torch.Size([1, 3, 512, 512])
        # x_cat: torch.Size([1, 16, 512, 512])

        return x_att

