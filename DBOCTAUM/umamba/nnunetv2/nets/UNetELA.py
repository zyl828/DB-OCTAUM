import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn

''''ELA'''

class ELA(nn.Module):
    def __init__(self, channel, kernel_size=7, groups=16, device='cuda'):
        super(ELA, self).__init__()
        self.device = device
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel,
                              bias=False).to(device)
        self.gn = nn.GroupNorm(groups, channel).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # 处理高度维度
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)

        # 处理宽度维度
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        return x * x_h * x_w



class MELA(nn.Module):
    def __init__(self, dim, group_kernel_sizes=[3, 5, 7, 9], device='cpu'):
        super(MELA, self).__init__()
        self.device = device
        self.dim = dim
        self.group_kernel_sizes = group_kernel_sizes


        assert self.dim // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.sigmoid = nn.Sigmoid()

        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b, c, h, w = x.size()
        # (B, C, H)
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # (B, C, W)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.sigmoid(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h, 1)

        x_w_attn = self.sigmoid(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w)
        x_attn = x * x_h_attn * x_w_attn

        return x_attn


class MSELA(nn.Module):
    def __init__(self, in_channels, factor=4.0):
        super(MSELA, self).__init__()
        dim = int(in_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.up = nn.Conv2d(dim, in_channels, kernel_size=1, stride=1)
        self.ela = ELA(dim, groups=8, device='cpu')
        self.relu = nn.ReLU()
        self.norm = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        x_fused = self.down(x)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x = self.relu(self.norm(x_fused_s))
        x = self.ela(x)
        x = self.up(x)

        return x

class Conv3ELA(nn.Module):
    def __init__(self, in_channels, factor=4.0):
        super(Conv3ELA, self).__init__()
        dim = int(in_channels // factor)
        self.conv_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.ela = ELA(in_channels, device='cpu')
        self.relu = nn.ReLU()
        self.norm = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        x = self.conv_3x3(x)
        # x = self.conv_1x1(x)
        x = self.norm(x)
        x = self.relu(x)
        # x = self.conv_3x3(x)
        x = self.conv_1x1(x)
        x = self.ela(x)

        return x

if __name__ == "__main__":
    x = torch.randn(4, 512, 32, 32)
    attn = Conv3ELA(512)
    y = attn(x)
    print(y.shape)
