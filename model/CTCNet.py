# 人脸，unet结构中间加线的模型
import torch
import torch.nn as nn

import numpy as np
import math
import torch.nn.functional as F
import numbers
from einops import rearrange


def make_model(opt):
    return CTCNet()


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class CTCNet(nn.Module):
    def __init__(self, conv=default_conv, norm_layer=nn.LayerNorm):
        super(CTCNet, self).__init__()

        self.scale = 8
        self.phase = 3
        n_blocks = 40
        n_feats = 20
        kernel_size = 3
        in_size = 96
        out_size = 96
        min_feat_size = 16
        res_depth = 10
        relu_type = 'leakyrelu'
        norm_type = 'bn'
        att_name = 'spar'
        bottleneck_size = 4

        self.head = conv(10, 32, kernel_size)

        hg_depth = 2

        ############ residual layers ############
        self.res_layers_down1_pre = ResidualBlock1_en_pre(32, 32, hg_depth=4, att_name=att_name, n_feat=32)
        self.res_layers_down1 = ResidualBlock1_en(64, 64, hg_depth=4, att_name=att_name, n_feat=64)
        self.res_layers_down2 = ResidualBlock2_en(128, 128, hg_depth=4, att_name=att_name, n_feat=128)

        self.res_layers_up1 = ResidualBlock1_de(128, 128, hg_depth=4, att_name=att_name, n_feat=128)
        self.res_layers_up2 = ResidualBlock2_de(64, 64, hg_depth=4, att_name=att_name, n_feat=64)

        self.res_layers = []
        for i in range(4):
            self.res_layers.append(ResidualBlock_res(128, 128, hg_depth=2, att_name=att_name, n_feat=128))
        self.res_layers = nn.Sequential(*self.res_layers)

        self.res_layers_back = ResidualBlock_res_back(32, 32, hg_depth=4, att_name=att_name, n_feat=32)

        self.tail = conv(32, 10, kernel_size)
        self.tail352_128 = conv(352, 128, kernel_size)
        self.tail288_64 = conv(288, 64, kernel_size)
        self.tai256_32 = conv(256, 32, kernel_size)
        self.conv_64 = nn.Conv2d(64, 64, 3, 2, 1)
        self.conv_32 = nn.Conv2d(32, 32, 3, 2, 1)
        self.tranpose_128 = nn.ConvTranspose2d(128, 128, 8, 2, 3)

        self.tranpose64_128 = nn.ConvTranspose2d(64, 64, 8, 2, 3)
        # self.tranpose128_128 = nn.ConvTranspose2d(128,128,8,4,2)
        self.tranpose128_128 = nn.ConvTranspose2d(128, 128, 6, 2, 2)

        self.down_input = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.LeakyReLU(0.2), )

        self.up_input = nn.Sequential(nn.ConvTranspose2d(32, 64, kernel_size=6, stride=2, padding=2, bias=False),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.LeakyReLU(0.2), )

        self.down1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.LeakyReLU(0.2), )

        self.down2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.LeakyReLU(0.2), )

        self.down3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.LeakyReLU(0.2), )

        self.up1 = nn.Sequential(nn.ConvTranspose2d(128, 128, kernel_size=6, stride=2, padding=2, bias=False),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.LeakyReLU(0.2), )

        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2, padding=2, bias=False),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.LeakyReLU(0.2), )

        self.up3 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=2, bias=False),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.LeakyReLU(0.2), )

        self.scale1_1 = Scale(0.25)
        self.scale1_2 = Scale(0.25)
        self.scale1_3 = Scale(0.25)
        self.scale1_4 = Scale(0.25)

        self.scale2_1 = Scale(0.25)
        self.scale2_2 = Scale(0.25)
        self.scale2_3 = Scale(0.25)
        self.scale2_4 = Scale(0.25)

        self.scale3_1 = Scale(0.25)
        self.scale3_2 = Scale(0.25)
        self.scale3_3 = Scale(0.25)
        self.scale3_4 = Scale(0.25)
        self.embed = nn.Conv2d(32, 32, 3, 1, 1)

        self.channel_func_128 = CALayer_128(16)
        self.channel_func_64 = CALayer_64(16)
        self.channel_func_32 = CALayer_32(16)

    def forward(self, x):  # x(1,10,96,96)

        input = x  # input(1,10,96,96)

        x = self.head(x)  # x(1,32,96,96)

        x = self.res_layers_down1_pre(x)  # x(1,32,96,96)

        x1 = self.down1(x)  # (1,64,48,48)
        x1 = self.res_layers_down1(x1)  # (1,64,48,48)

        x2 = self.down2(x1)  # (16,128,32,32)
        x2 = self.res_layers_down2(x2)  # (1,128,24,24)

        x3 = self.down3(x2)  # (1,128,12,12)

        res1 = x  # (1,32,96,96)

        z = self.res_layers(x3)  # (1,128,12,12)

        y1 = self.up1(z)  # (1,128,12,12)
        y1_1 = self.conv_64(x1)  # (1,64,24,24)
        y1_2 = self.conv_32(x)  # (1,32,48,48)
        y1_2 = self.conv_32(y1_2)  # (1,32,24,24)

        y1 = torch.cat([y1, x2, y1_1, y1_2], 1)  # y1(1,352,24,24)
        y1 = self.tail352_128(y1)  # y1(1,128,24,24)

        y1 = self.channel_func_128(y1)  # y1(1,128,24,24)

        y1 = self.res_layers_up1(y1)  # (1,128,24,24)

        y2 = self.up2(y1)  # (1,64,48,48)
        y2_1 = self.conv_32(x)  # (1,32,48,48)
        y2_2 = self.tranpose_128(x2)  # (1,128,48,48)

        y2 = torch.cat([y2, x1, y2_1, y2_2], 1)  # (32,288,64,64)
        y2 = self.tail288_64(y2)  # (32,64,64,64)

        y2 = self.channel_func_64(y2)

        y2 = self.res_layers_up2(y2)  # (32,64,64,64)

        ######## Third #####
        y3 = self.up3(y2)  # (1,32,128,128)

        y3_1 = self.tranpose64_128(x1)  # (32,64,128,128)

        y3_2 = self.tranpose128_128(x2)  # (32,128,128,128)
        y3_2 = self.tranpose128_128(y3_2)

        y3 = torch.cat([y3, x, y3_1, y3_2], 1)  # (32,256,128,128)
        y3 = self.tai256_32(y3)  # (32,32,128,128)

        y3 = self.channel_func_32(y3)

        y3 = self.res_layers_back(y3)

        out = self.tail(y3)  # (32,3,128,128)
        out = out + input

        return out


class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ReluLayer(nn.Module):

    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x
        else:
            assert 1 == 0, 'Relu type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)


class NormLayer(nn.Module):

    def __init__(self, channels, normalize_shape=None, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x
        else:
            assert 1 == 0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class CALayer_128(nn.Module):
    def __init__(self, reduction=16):
        super(CALayer_128, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(128, 128 // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128 // reduction, 128, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):  # (32,64,32,32)

        y = self.avg_pool(x)  # (32,64,1,1)
        y = self.conv_du(y)  # (32,64,1,1)
        return x * y  # (32,64,32,32)


class CALayer_64(nn.Module):
    def __init__(self, reduction=16):
        super(CALayer_64, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(64, 64 // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 // reduction, 64, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):  # (32,64,32,32)

        y = self.avg_pool(x)  # (32,64,1,1)
        y = self.conv_du(y)  # (32,64,1,1)
        return x * y  # (32,64,32,32)


class CALayer_32(nn.Module):
    def __init__(self, reduction=16):
        super(CALayer_32, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(32, 32 // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32 // reduction, 32, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):  # (32,64,32,32)

        y = self.avg_pool(x)  # (32,64,1,1)
        y = self.conv_du(y)  # (32,64,1,1)
        return x * y  # (32,64,32,32)


class HourGlassBlock(nn.Module):

    def __init__(self, n_feat, depth, c_in, c_out, c_mid=64, norm_type='bn', relu_type='prelu', ):
        super(HourGlassBlock, self).__init__()
        self.depth = depth
        self.c_in = c_in
        self.c_mid = c_mid
        self.c_out = c_out
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}
        if self.depth:
            self._generate_network(self.depth)
            self.out_block = nn.Sequential(
                ConvLayer(self.c_mid, self.c_out, norm_type='none', relu_type='none'),
                nn.Sigmoid()
            )

        self.channel_func = CALayer(n_feat, 16)

    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs))
        self.add_module('b2_' + str(level), ConvLayer(c1, c2, scale='down', **self.kwargs))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid, **self.kwargs))

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', **self.kwargs))

    def _forward(self, level, in_x):
        up1 = self._modules['b1_' + str(level)](in_x)
        low1 = self._modules['b2_' + str(level)](in_x)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        up2 = self._modules['b3_' + str(level)](low2)
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])

        return up1 + up2

    def forward(self, x, pmask=None):
        if self.depth == 0: return x
        input_x = x  # (32,64,64,64)

        x = self._forward(self.depth, x)  # (32,64,64,64)

        x = self.channel_func(x)  # (32,64,64,64)

        x = x + input_x  # (32,64,64,64)
        self.att_map = self.out_block(x)  # (32,1,64,64)
        x = input_x * self.att_map  # (32,64,64,64)
        return x


class HourGlassBlock2(nn.Module):
    def __init__(self, n_feat, depth, c_in, c_out, c_mid=64, norm_type='bn', relu_type='prelu', ):
        super(HourGlassBlock2, self).__init__()
        self.depth = depth
        self.c_in = c_in
        self.c_mid = c_mid
        self.c_out = c_out
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}
        if self.depth:
            self._generate_network(self.depth)
            self.out_block = nn.Sequential(
                ConvLayer(self.c_mid, self.c_out, norm_type='none', relu_type='none'),
                nn.Sigmoid()
            )

        self.channel_func = CALayer2(n_feat, 16)
        self._64 = nn.Conv2d(128, 64, 1)

    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs))
        self.add_module('b2_' + str(level), ConvLayer(c1, c2, scale='down', **self.kwargs))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid, **self.kwargs))

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', **self.kwargs))

    def _forward(self, level, in_x):
        up1 = self._modules['b1_' + str(level)](in_x)
        low1 = self._modules['b2_' + str(level)](in_x)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        up2 = self._modules['b3_' + str(level)](low2)
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])

        return up1 + up2

    def forward(self, x, pmask=None):  # (32,128,32,32)
        if self.depth == 0: return x

        input_x = x  # (32,128,32,32)

        plus = self._64(x)  # (32,64,32,32)
        x = self._forward(self.depth, x)  # (32,64,32,32)

        x = self.channel_func(x)  # (32,64,32,32)

        x = x + plus  # (32,64,64,64)
        self.att_map = self.out_block(x)  # (32,1,64,64)

        x = input_x * self.att_map  # (32,64,64,64)
        return x


class HourGlassBlock32(nn.Module):
    # HourGlassBlock32(self.n_feat, self.hg_depth, c_out, c_attn, **kwargs)
    def __init__(self, n_feat, depth, c_in, c_attn, norm_type='bn', relu_type='prelu', ):
        super(HourGlassBlock32, self).__init__()
        self.depth = depth
        self.c_in = c_in
        self.c_mid = n_feat*2
        self.c_attn = c_attn
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}
        if self.depth:
            self._generate_network(self.depth)
            self.out_block = nn.Sequential(
                ConvLayer(self.c_mid, self.c_attn, norm_type='none', relu_type='none'),
                nn.Sigmoid()
            )

        self.channel_func = CALayer2(n_feat, n_feat//2)
        self._64 = nn.Conv2d(n_feat, n_feat*2, 1)

    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs))
        self.add_module('b2_' + str(level), ConvLayer(c1, c2, scale='down', **self.kwargs))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid, **self.kwargs))

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', **self.kwargs))

    def _forward(self, level, in_x):
        up1 = self._modules['b1_' + str(level)](in_x)
        low1 = self._modules['b2_' + str(level)](in_x)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        up2 = self._modules['b3_' + str(level)](low2)
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])

        return up1 + up2

    def forward(self, x):  # (1,32,96,96)
        if self.depth == 0: return x

        input_x = x  # (1,32,96,96)

        plus = self._64(x)  # (1,64,96,96)
        x = self._forward(self.depth, x)  # (1,64,96,96)

        x = self.channel_func(x)  # (1,64,96,96)

        x = x + plus  # (1,64,96,96)
        self.att_map = self.out_block(x)  # (1,1,96,96)

        x = input_x * self.att_map  # (1,32,96,96)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale='none', norm_type='none', relu_type='none',
                 use_pad=True):
        super(ConvLayer, self).__init__()
        self.use_pad = use_pad

        bias = True if norm_type in ['pixel', 'none'] else False
        stride = 2 if scale == 'down' else 1

        self.scale_func = lambda x: x
        if scale == 'up':
            self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.relu = ReluLayer(out_channels, relu_type)
        self.norm = NormLayer(out_channels, norm_type=norm_type)

    def forward(self, x):
        out = self.scale_func(x)
        if self.use_pad:
            out = self.reflection_pad(out)
        out = self.conv2d(out)
        out = self.norm(out)
        out = self.relu(out)
        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, relu=True,
                 bn=False, bias=False, up_size=0, fan=False):
        super(BasicConv, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.out_channels = out_planes
        self.in_channels = in_planes
        if fan:
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                           padding=padding,
                                           dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x


class one_conv(nn.Module):
    def __init__(self, inchanels, growth_rate, kernel_size=3, relu=True):
        super(one_conv, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.conv = nn.Conv2d(inchanels, growth_rate, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        self.conv1 = nn.Conv2d(growth_rate, inchanels, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        if relu:
            self.relu = nn.PReLU(growth_rate)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)

    def forward(self, x):
        output = self.weight1(x) + self.weight2(self.conv1(self.relu(self.conv(x))))
        return output


class new_CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(new_CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class one_module(nn.Module):
    def __init__(self, n_feats):
        super(one_module, self).__init__()
        self.layer1 = one_conv(n_feats, n_feats // 2, 3)
        self.layer2 = one_conv(n_feats, n_feats // 2, 3)
        self.layer4 = BasicConv(n_feats, n_feats, 3, 1, 1)
        self.alise = BasicConv(2 * n_feats, n_feats, 1, 1, 0)
        self.atten = CALayer(n_feats)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)
        self.weight3 = Scale(1)
        self.weight4 = Scale(1)
        self.weight5 = Scale(1)

    def forward(self, x):  # x(1,32,96,96)
        x1 = self.layer1(x)  # x(1,32,96,96)
        x2 = self.layer2(x1)
        x4 = self.layer4(self.atten(self.alise(torch.cat([self.weight2(x2), self.weight3(x1)], 1))))
        return self.weight4(x) + self.weight5(x4)


class ResidualBlock1_en_pre(nn.Module):
    # ResidualBlock1_en_pre(32, 32, hg_depth=4, att_name=att_name, n_feat=32)
    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock1_en_pre, self).__init__()
        self.c_in = c_in  # 32
        self.c_out = c_out  # 32
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        self.encoder = one_module(n_feat)
        self.alise = one_module(n_feat)
        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func = HourGlassBlock32(self.n_feat, self.hg_depth, c_in, c_attn, **kwargs)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
              for i in range(4)])

    def forward(self, x):  # x(1,32,96,96)
        a = x.shape[0]

        identity = self.shortcut_func(x)  # identity(1,32,96,96)

        x = self.encoder(x)  # (1,32,96,96)

        x = self.att_func(x)  # (1,32,96,96)

        x = self.alise(x)
        out = self.encoder_level1(x)
        out = out + identity

        return out


class ResidualBlock1_en(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock1_en, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        # discrimintor
        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        self.encoder = one_module(n_feat)
        self.alise = one_module(n_feat)
        self.att = new_CALayer(n_feat)

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func = HourGlassBlock(self.n_feat, self.hg_depth, c_out, c_attn, **kwargs)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=64, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
              for i in range(6)])

    def forward(self, x):  # (32,64,64,64)
        a = x.shape[0]

        identity = self.shortcut_func(x)  # (32,64,64,64)

        x = self.encoder(x)  # (32,64,64,64)

        x = self.att_func(x)  # (32,64,64,64)

        x = self.alise(x)
        out = self.encoder_level1(x)
        out = out + identity

        return out


class ResidualBlock2_en(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock2_en, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        # discrimintor
        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        self.encoder = one_module(n_feat)
        self.alise = one_module(n_feat)
        self.att = new_CALayer(n_feat)

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func2 = HourGlassBlock2(self.n_feat, self.hg_depth, c_out, c_attn, **kwargs)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=128, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
              for i in range(6)])

    def forward(self, x):  # (32,128,32,32)

        a = x.shape[0]
        identity = self.shortcut_func(x)  # (32,128,32,32)
        x = self.encoder(x)

        x = self.att_func2(x)  # (32,64,64,64)

        x = self.alise(x)
        out = self.encoder_level1(x)

        out = out + identity

        return out


class ResidualBlock1_de(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock1_de, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        # discrimintor
        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        self.encoder = one_module(n_feat)
        self.alise = one_module(n_feat)
        self.att = new_CALayer(n_feat)

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func2 = HourGlassBlock2(self.n_feat, self.hg_depth, c_out, c_attn, **kwargs)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=128, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
              for i in range(6)])

    def forward(self, x):  # (32,128,32,32)
        a = x.shape[0]
        identity = self.shortcut_func(x)  # (32,128,32,32)
        x = self.encoder(x)

        x = self.att_func2(x)  # (32,64,64,64)

        x = self.alise(x)
        out = self.encoder_level1(x)

        out = out + identity

        return out


class ResidualBlock2_de(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock2_de, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        # discrimintor
        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        self.encoder = one_module(n_feat)
        self.alise = one_module(n_feat)
        self.att = new_CALayer(n_feat)

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func = HourGlassBlock(self.n_feat, self.hg_depth, c_out, c_attn, **kwargs)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=64, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
              for i in range(6)])

    def forward(self, x):  # (32,128,32,32)
        a = x.shape[0]
        identity = self.shortcut_func(x)  # (32,128,32,32)
        x = self.encoder(x)

        x = self.att_func(x)  # (32,64,64,64)

        x = self.alise(x)
        out = self.encoder_level1(x)

        out = out + identity

        return out


class ResidualBlock_res(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock_res, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        # discrimintor
        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        self.encoder = one_module(n_feat)
        self.STL64 = SwinTransformerBlock(dim=64, drop=0., attn_drop=0, drop_path=0, norm_layer=nn.LayerNorm)
        self.STL128 = SwinTransformerBlock(dim=128, drop=0., attn_drop=0, drop_path=0, norm_layer=nn.LayerNorm)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.decoder_low = one_module(n_feat)
        self.decoder_high = one_module(n_feat)
        self.alise = one_module(n_feat)
        self.alise2 = BasicConv(2 * n_feat, n_feat, 1, 1, 0)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = new_CALayer(n_feat)

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        self.conv1 = ConvLayer(c_in, c_out, 3, scales[0], **kwargs)
        self.conv2 = ConvLayer(c_out, c_out, 3, scales[1], norm_type=norm_type, relu_type='none')

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func2 = HourGlassBlock2(self.n_feat, self.hg_depth, c_out, c_attn, **kwargs)
        self.scale_1 = Scale(0.5)
        self.scale_2 = Scale(0.5)
        self.sigmoid = nn.Sigmoid()
        self.compress1 = nn.Conv2d(256, 128, 1)
        self.compress2 = nn.Conv2d(128, 128, 1)

    def forward(self, x):  # (32,128,32,32)

        a = x.shape[0]
        a = x.shape[0]  # 32
        identity = self.shortcut_func(x)  # (32,64,64,64)
        x = self.encoder(x)

        x = self.att_func2(x)  # (32,64,64,64)
        x = self.alise(x)

        x2 = self.down(x)  # (32,64,32,32)
        high = x - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)  # (32,64,64,64)
        x2 = self.decoder_low(x2)  # (32,64,32,32))
        x3 = x2  # (32,64,32,32)

        high1 = self.decoder_high(high)  # (32,64,64,64)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)  # (32,64,64,64)

        x4 = self.scale_1(x4)
        high1 = self.scale_2(high1)
        concat1 = torch.cat([x4, high1], dim=1)  # (32,128,64,64)

        concat1 = self.compress1(concat1)

        y1 = self.decoder_high(concat1)  # (32,64,64,64)
        y2 = self.decoder_low(x2)  # (32,64,32,32)
        y2 = F.interpolate(y2, size=x.size()[-2:], mode='bilinear', align_corners=True)  # (32,64,64,64)
        y1 = self.scale_1(y1)
        y2 = self.scale_2(y2)
        sig = self.compress2(x)
        x_sig = x * sig
        out = self.alise2(torch.cat([y2, y1], dim=1)) + x_sig
        out = self.alise(out)
        out = out + identity

        return out


class ResidualBlock_res_back(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock_res_back, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        # discrimintor
        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        self.encoder = one_module(n_feat)
        self.alise = one_module(n_feat)
        self.att = new_CALayer(n_feat)

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func2 = HourGlassBlock32(self.n_feat, self.hg_depth, c_out, c_attn, **kwargs)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
              for i in range(4)])

    def forward(self, x):  # (32,128,32,32)

        identity = self.shortcut_func(x)  # (32,64,64,64)
        x = self.encoder(x)

        x = self.att_func2(x)  # (32,64,64,64)

        out = self.alise(x)
        out = self.encoder_level1(out)
        out = out + identity

        return out


class CALayer2(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(64, 64 // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 // reduction, 64, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):  # (32,64,32,32)

        y = self.avg_pool(x)  # (32,64,1,1)
        y = self.conv_du(y)  # (32,64,1,1)
        return x * y  # (32,64,32,32)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)  # (1,64,1,1)
        y = self.conv_du(y)  # (1,64,1,1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, num_in, num_out, need_bn=True):
        super(ResidualBlock, self).__init__()
        if need_bn:
            self.conv_block = nn.Sequential(
                nn.Conv2d(num_in, num_out // 2, 1),
                nn.BatchNorm2d(num_out // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_out // 2, num_out // 2, 3, stride=1, padding=1),
                nn.BatchNorm2d(num_out // 2), nn.ReLU(inplace=True),
                nn.Conv2d(num_out // 2, num_out, 1), nn.BatchNorm2d(num_out))
            self.skip_layer = None if num_in == num_out else nn.Sequential(
                nn.Conv2d(num_in, num_out, 1), nn.BatchNorm2d(num_out))
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(num_in, num_out // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_out // 2, num_out // 2, 3, stride=1, padding=1),
                nn.Conv2d(num_out // 2, num_out, 1))
            self.skip_layer = None if num_in == num_out else nn.Conv2d(num_in, num_out, 1)

    def forward(self, x):
        residual = self.conv_block(x)
        if self.skip_layer:
            x = self.skip_layer(x)
        return x + residual


class HourGlass(nn.Module):
    def __init__(self, num_layer, num_feature, need_bn=True):
        super(HourGlass, self).__init__()
        self._n = num_layer
        self._f = num_feature
        self.need_bn = need_bn
        self._init_layers(self._n, self._f)

    def _init_layers(self, n, f):
        setattr(self, 'res' + str(n) + '_1', ResidualBlock(f, f, self.need_bn))
        setattr(self, 'pool' + str(n) + '_1', nn.MaxPool2d(2, 2))
        setattr(self, 'res' + str(n) + '_2', ResidualBlock(f, f, self.need_bn))
        if n > 1:
            self._init_layers(n - 1, f)
        else:
            self.res_center = ResidualBlock(f, f, self.need_bn)
        setattr(self, 'res' + str(n) + '_3', ResidualBlock(f, f, self.need_bn))

    def _forward(self, x, n, f):
        up1 = eval('self.res' + str(n) + '_1')(x)

        low1 = eval('self.pool' + str(n) + '_1')(x)
        low1 = eval('self.res' + str(n) + '_2')(low1)
        if n > 1:
            low2 = self._forward(low1, n - 1, f)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.' + 'res' + str(n) + '_3')(low3)
        up2 = nn.functional.interpolate(low3, scale_factor=2, mode='bilinear', align_corners=True)

        return up1 + up2

    def forward(self, x):
        return self._forward(x, self._n, self._f)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):

        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * 4)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)  # (32,1024,128)
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # (32,1024,128)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out





class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


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


class DownBlock(nn.Module):
    def __init__(self, scale, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()

        if nFeat is None:
            nFeat = 20

        if in_channels is None:
            in_channels = 3

        if out_channels is None:
            out_channels = 3

        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )
        ]

        # 该模块在本代码中不起作用，scale默认均为2
        for _ in range(1, int(np.log2(
                scale))):  # 当scale=2时，此循环不起作用，dual_block仅包括conv-LeakyReLU-conv;当scale=4时，此循环起作用，dual_block包括conv-LeakyReLU-conv-LeakyReLU-conv;
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.ReLU(inplace=True)
                )
            )

        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x


class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class EcaLayer(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super(EcaLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class MSRB(nn.Module):
    def __init__(self, conv, n_feat):
        super(MSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5
        # self.ca1 = EcaLayer(n_feat)
        # self.ca2 = EcaLayer(n_feat*2)
        # self.ca3 = EcaLayer(n_feat * 4)

        self.ca1 = CALayer(channel=n_feat)
        self.ca2 = CALayer(channel=n_feat * 2)
        self.ca3 = CALayer(channel=n_feat * 4)

        self.conv_3_1 = conv(n_feat, n_feat, kernel_size_1)
        self.conv_3_2 = conv(n_feat * 2, n_feat * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feat, n_feat, kernel_size_2)
        self.conv_5_2 = conv(n_feat * 2, n_feat * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feat * 4, n_feat, 1, padding=0, stride=1)
        self.confusion1 = nn.Conv2d(n_feat * 4, n_feat * 4, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.conv_3_1(self.relu(self.conv_3_1(input_1)))
        output_5_1 = self.conv_5_1(self.relu(self.conv_5_1(input_1)))

        output_ca_31 = self.ca1(output_3_1)
        output_ca_51 = self.ca1(output_5_1)

        output_ca_31_1 = input_1 + output_ca_31
        output_ca_51_1 = input_1 + output_ca_51

        input_2 = torch.cat([output_ca_31_1, output_ca_51_1], 1)

        output_3_2 = self.conv_3_2(self.relu(self.conv_3_2(input_2)))
        output_5_2 = self.conv_5_2(self.relu(self.conv_5_2(input_2)))

        output_ca_32 = self.ca2(output_3_2)
        output_ca_52 = self.ca2(output_5_2)

        output_ca_32_2 = input_2 + output_ca_32
        output_ca_52_2 = input_2 + output_ca_52

        input_3 = torch.cat([output_ca_32_2, output_ca_52_2], 1)

        output = self.confusion1(self.relu(self.confusion1(input_3)))
        output_4 = self.ca3(output)
        output5 = input_3 + output_4

        output6 = self.confusion(output5)
        output6 += x
        return output6


class RCAB_ECA(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB_ECA, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(EcaLayer(channels=n_feat))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res