import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class branch_block_2d(nn.Module):
    def __init__(self, in_channel=32, out_channel=32, kernel_size=(1, 3, 3), padding=(0, 1, 1)):
        super(branch_block_2d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1,
                              padding=padding, bias=True)
        self.PRelu = nn.PReLU()

    def forward(self, x):
        return self.PRelu(self.conv(x))


class FE_block_2d(nn.Module):
    def __init__(self, in_channel=32):
        super(FE_block_2d, self).__init__()
        self.in_channel = in_channel
        assert in_channel % 4 == 0
        self.out_channel = in_channel // 4

        self.uv_conv = branch_block_2d(in_channel=self.in_channel, out_channel=self.out_channel, kernel_size=(3, 3, 1),
                                       padding=(1, 1, 0))
        self.xy_conv = branch_block_2d(in_channel=self.in_channel, out_channel=self.out_channel, kernel_size=(1, 3, 3),
                                       padding=(0, 1, 1))
        self.ux_conv = branch_block_2d(in_channel=self.in_channel, out_channel=self.out_channel, kernel_size=(3, 3, 1),
                                       padding=(1, 1, 0))
        self.vy_conv = branch_block_2d(in_channel=self.in_channel, out_channel=self.out_channel, kernel_size=(1, 3, 3),
                                       padding=(0, 1, 1))

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = list(x.shape)

        x = x.reshape(batch, channel, height_view, width_view, height * width)
        uv_data = self.uv_conv(x)
        uv_data = uv_data.reshape(batch, channel // 4, height_view, width_view, height, width)

        x = x.reshape(batch, channel, height_view * width_view, height, width)
        xy_data = self.xy_conv(x)
        xy_data = xy_data.reshape(batch, channel // 4, height_view, width_view, height, width)

        x = x.reshape(batch, channel, height_view, width_view, height, width)
        x = x.permute(0, 1, 2, 4, 3, 5)

        x = x.reshape(batch, channel, height_view, height, width_view * width)
        ux_data = self.ux_conv(x)
        ux_data = ux_data.reshape(batch, channel // 4, height_view, height, width_view, width)
        ux_data = ux_data.permute(0, 1, 2, 4, 3, 5)

        x = x.reshape(batch, channel, height_view * height, width_view, width)
        vy_data = self.vy_conv(x)
        vy_data = vy_data.reshape(batch, channel // 4, height_view, height, width_view, width)
        vy_data = vy_data.permute(0, 1, 2, 4, 3, 5)
        del x

        return torch.cat([uv_data, xy_data, ux_data, vy_data], dim=1)


class branch_block_3d(nn.Module):
    def __init__(self, in_channel=32, out_channel=32, kernel_size=3, padding=1):
        super(branch_block_3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1,
                              padding=padding, bias=True)
        self.PRelu = nn.PReLU()

    def forward(self, x):
        return self.PRelu(self.conv(x))


class FE_block_3d(nn.Module):
    def __init__(self, in_channel=32):
        super(FE_block_3d, self).__init__()
        self.in_channel = in_channel
        assert in_channel % 4 == 0
        self.out_channel = in_channel // 4

        self.uvx_conv = branch_block_3d(in_channel=self.in_channel, out_channel=self.out_channel)
        self.uvy_conv = branch_block_3d(in_channel=self.in_channel, out_channel=self.out_channel)
        self.uxy_conv = branch_block_3d(in_channel=self.in_channel, out_channel=self.out_channel)
        self.vxy_conv = branch_block_3d(in_channel=self.in_channel, out_channel=self.out_channel)

    def forward(self, x):
        batch, channel, height_view, width_view, height, width = list(x.shape)
        channel = channel // 4

        uvx_data = torch.zeros(batch, channel, height_view, width_view, height, width).cuda()
        for i in range(width):
            uvx_data[:, :, :, :, :, i] = self.uvx_conv(x[:, :, :, :, :, i])

        uvy_data = torch.zeros(batch, channel, height_view, width_view, height, width).cuda()
        for i in range(height):
            uvy_data[:, :, :, :, i, :] = self.uvy_conv(x[:, :, :, :, i, :])

        uxy_data = torch.zeros(batch, channel, height_view, width_view, height, width).cuda()
        for i in range(width_view):
            uxy_data[:, :, :, i, :, :] = self.uxy_conv(x[:, :, :, i, :, :])

        vxy_data = torch.zeros(batch, channel, height_view, width_view, height, width).cuda()
        for i in range(height_view):
            vxy_data[:, :, i, :, :, :] = self.vxy_conv(x[:, :, i, :, :, :])

        return torch.cat([uvx_data, uvy_data, uxy_data, vxy_data], dim=1)


class res_block_2d(nn.Module):
    def __init__(self, channel):
        super(res_block_2d, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.PRelu = nn.PReLU()

    def forward(self, x):
        x = x + self.PRelu(self.conv(x))
        return x


class ResBlock3d(nn.Module):
    def __init__(self, n_feats, kernel_size, padding=(2, 1, 1), bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock3d, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv3d(n_feats, n_feats, kernel_size, padding=padding, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class ResBlockc3d(nn.Module):
    def __init__(self, n_feats, bias=True, res_scale=1, is_large_kernel=False):
        super(ResBlockc3d, self).__init__()
        if is_large_kernel:
            kernel_size = (5, 3, 3)
            padding = (2, 1, 1)
        else:
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)
        m = []

        m.append(nn.PReLU())
        # m.append(nn.ReLU())
        # m.append(nn.Conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=bias))
        # m.append(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=bias))

        m.append(nn.Conv3d(n_feats, n_feats, kernel_size=kernel_size, padding=padding, bias=bias))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res


class ResBlock_SA_2d(nn.Module):
    def __init__(self, n_feats, padding=1, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock_SA_2d, self).__init__()
        m = []
        m.append(nn.PReLU())
        m.append(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=bias))

        n = []
        n.append(nn.PReLU())
        n.append(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=bias))

        self.s_body = nn.Sequential(*m)
        self.a_body = nn.Sequential(*n)
        self.res_scale = res_scale
        self.view_n = None
        self.image_h = None
        self.image_w = None
        self.n_feats = n_feats

    def forward(self, x):
        res = self.s_body(x)
        # res = res.view(-1, self.n_feats, self.view_n, self.view_n, self.image_h, self.image_w)
        res = res.permute(0, 1, 3, 4, 2)
        res = res.view(-1, self.n_feats, self.image_h * self.image_w, self.view_n, self.view_n)
        res = self.a_body(res)
        # res = res.view(-1, self.n_feats, self.image_h, self.image_w, self.view_n, self.view_n, )
        res = res.permute(0, 1, 3, 4, 2)
        res = res.view(-1, self.n_feats, self.view_n * self.view_n, self.image_h, self.image_w)
        res += x

        return res


class ResBlock2d(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, padding=1, bias=True, bn=False, act=nn.ReLU(True),
                 res_scale=1):
        super(ResBlock2d, self).__init__()
        m = []
        m.append(nn.Conv3d(in_feats, out_feats, kernel_size, padding=[0, 1, 1], bias=bias))
        m.append(nn.PReLU())
        # m.append(nn.ReLU())

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)

        return res


class Block2d(nn.Module):
    def __init__(self, n_feats, kernel_size, padding=1, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(Block2d, self).__init__()
        m = []
        m.append(nn.Conv3d(n_feats, n_feats // 2, kernel_size, padding=[0, 1, 1], bias=bias))
        m.append(nn.PReLU())

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
