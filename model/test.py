import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import numbers
from einops import rearrange

def make_model(opt):
    return Restormer(opt=opt)


class Restormer(nn.Module):
    def __init__(self,
                 opt,
                 inp_channels=10,
                 out_channels=10,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 att_name='spar'
                 ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed() # 改

        self.layers1 = nn.Sequential(ResidualBlock(dim, dim, hg_depth=4, att_name=att_name, n_feat=dim))
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.layers2 = nn.Sequential(ResidualBlock(int(dim * 2 ** 1), int(dim * 2 ** 1), hg_depth=4, att_name=att_name, n_feat=int(dim * 2 ** 1)))
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.layers3 = nn.Sequential(ResidualBlock(int(dim * 2 ** 2), int(dim * 2 ** 2), hg_depth=4, att_name=att_name, n_feat=int(dim * 2 ** 2)))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.layers1(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(out_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.layers2(inp_enc_level2)
        out_enc_level2 = self.encoder_level2(out_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.layers3(inp_enc_level3)
        out_enc_level3 = self.encoder_level3(out_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1






##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
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


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):  # x(48,96,96)
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))  # qkv(144,96,96)
        q, k, v = qkv.chunk(3, dim=1)  # q(48,96,96)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # q(1,48,9216)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # k(1,48,9216)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # v(1,48,9216)

        q = torch.nn.functional.normalize(q, dim=-1)  # q(1,48,9216)
        k = torch.nn.functional.normalize(k, dim=-1)  # k(1,48,9216)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # attn(1,48,48)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)  # out(1,48,9216)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
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


##########################################################################
## Overlapped image patch embedding with 3x3 Conv 用3x3卷积的重叠图像块嵌入
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=10, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class ResidualBlock(nn.Module):

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, n_feat=20,
                 att_name='spar'):
        super(ResidualBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth
        self.n_feat = n_feat

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        self.encoder = one_module(n_feat)
        self.alise = one_module(n_feat)
        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func = HourGlassBlock(self.n_feat, self.hg_depth, c_in, c_attn, **kwargs)


    def forward(self, x):  # (1,48,96,96)
        x = self.encoder(x)  # (32,32,128,128)
        x = self.att_func(x)  # (32,64,64,64)
        x = self.alise(x)

        return x


class HourGlassBlock(nn.Module):

    def __init__(self, n_feat, depth, c_in, c_attn, norm_type='bn', relu_type='prelu', ):
        super(HourGlassBlock, self).__init__()
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

        self.channel_func = CALayer2(self.c_mid, 16)
        self._64 = nn.Conv2d(n_feat, self.c_mid, 1)

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

    def forward(self, x):  # x(1,48,96,96)
        if self.depth == 0: return x

        input_x = x  # x(1,48,96,96)

        plus = self._64(x)  # (1,96,96,96)
        x = self._forward(self.depth, x)  # (1,96,96,96)

        x = self.channel_func(x)  # (1,96,96,96)

        x = x + plus  # (1,96,96,96)
        self.att_map = self.out_block(x)  # (1,1,96,96)

        x = input_x * self.att_map  # (1,48,96,96)
        return x


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


class CALayer2(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):  # (1,96,96,96)

        y = self.avg_pool(x)  # (1,96,1,1)
        y = self.conv_du(y)  # (1,96,1,1)
        return x * y


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

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x4 = self.layer4(self.atten(self.alise(torch.cat([self.weight2(x2), self.weight3(x1)], 1))))
        return self.weight4(x) + self.weight5(x4)


class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


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
        self.flag = relu
        self.conv1 = nn.Conv2d(growth_rate, inchanels, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        if relu:
            self.relu = nn.PReLU(growth_rate)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)

    def forward(self, x):
        if self.flag == False:
            output = self.weight1(x) + self.weight2(self.conv1(self.conv(x)))
        else:
            output = self.weight1(x) + self.weight2(self.conv1(self.relu(self.conv(x))))
        return output