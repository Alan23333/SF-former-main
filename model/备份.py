# 在CatCA上无CA，无LL、光谱分流,加超级光谱增强
import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import numbers
from einops import rearrange
import torch

from timm.models.layers import DropPath
from einops import rearrange


def make_model(opt):
    return CAT_Unet(opt=opt)


# The implementation builds on Restormer code https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py
class CAT_Unet(nn.Module):
    def __init__(self,
                 opt,
                 img_size=96,
                 in_chans=10,
                 dim=48,
                 depth=[1, 4, 6, 6, 8],
                 split_size_0=[4, 4, 4, 4, 4],
                 num_heads=[2, 2, 2, 4, 8],
                 mlp_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 img_range=1.,
                 bias=False,
                 num_refinement_blocks=4,
                 **kwargs
    ):

        super(CAT_Unet, self).__init__()

        out_channels = in_chans

        self.patch_embed = OverlapPatchEmbed(in_chans, dim)

        self.layer1 = nn.ModuleList([CATB_axial(dim=dim, num_heads=num_heads[0],
                                                reso=img_size,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                split_size=split_size_0[0],
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=drop_path_rate,
                                                act_layer=act_layer,
                                                norm_layer=norm_layer,
                                                shift_size=0 if (i % 2 == 0) else split_size_0[0] // 2,
                                                )
                                     for i in range(depth[0])])

        self.layer2 = nn.ModuleList([CATB_axial(dim=dim, num_heads=num_heads[0],
                                                reso=img_size,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                split_size=split_size_0[0],
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=drop_path_rate,
                                                act_layer=act_layer,
                                                norm_layer=norm_layer,
                                                shift_size=0 if (i % 2 == 0) else split_size_0[0] // 2,
                                                )
                                     for i in range(depth[0])])

        self.layer3 = nn.ModuleList([CATB_axial(dim=dim, num_heads=num_heads[0],
                                                reso=img_size,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                split_size=split_size_0[0],
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=drop_path_rate,
                                                act_layer=act_layer,
                                                norm_layer=norm_layer,
                                                shift_size=0 if (i % 2 == 0) else split_size_0[0] // 2,
                                                )
                                     for i in range(depth[0])])

        self.encoder_level1 = nn.ModuleList([CATB_axial(dim=dim,
                                                           num_heads=num_heads[1],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[1],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[1]//2,
                                                           )
                                            for i in range(depth[1])])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([CATB_axial(dim=int(dim*2**1),
                                                           num_heads=num_heads[2],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[2],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[2]//2,
                                                           )
                                            for i in range(depth[2])])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([CATB_axial(dim=int(dim*2**2),
                                                           num_heads=num_heads[3],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[3],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[3]//2,
                                                           )
                                            for i in range(depth[3])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.ModuleList([CATB_axial(dim=int(dim*2**3),
                                                           num_heads=num_heads[4],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[4],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[4]//2,
                                                           )
                                            for i in range(depth[4])])

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([CATB_axial(dim=int(dim*2**2),
                                                           num_heads=num_heads[3],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[3],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[3]//2,
                                                           )
                                            for i in range(depth[3])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([CATB_axial(dim=int(dim*2**1),
                                                           num_heads=num_heads[2],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[2],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[2]//2,
                                                           )
                                            for i in range(depth[2])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([CATB_axial(dim=int(dim*2**1),
                                                           num_heads=num_heads[1],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[1],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[1]//2,
                                                           )
                                            for i in range(depth[1])])

        self.refinement = nn.ModuleList([CATB_axial(dim=int(dim*2**1),
                                                           num_heads=num_heads[1],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[1],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[1]//2,
                                                           )
                                            for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.CC = CC_Module()


    def forward(self, inp_img):  # inp_img(10,96,96)
        _, _, H, W = inp_img.shape  # H=96 W=96
        inp_enc_level1 = self.patch_embed(inp_img)  # inp_enc_level1(1,9216,48) 把它按维度变成条条

        inp_enc_level1 = rearrange(inp_enc_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()  # layer2(1,48,96,96)

        inp_enc_level1 = self.CC(inp_enc_level1)



        inp_enc_level1 = rearrange(inp_enc_level1, "b c h w -> b (h w) c", h=H, w=W).contiguous()  # layer1(1,48,96,96)


        out_enc_level1 = inp_enc_level1  # out_enc_level1(9216,48)
        for layer in self.encoder_level1:  # 循环4次
            out_enc_level1 = layer(out_enc_level1, [H, W])  # out_enc_level1(9216,48)
        # out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1, H, W)  # inp_enc_level2(2304,96)
        out_enc_level2 = inp_enc_level2  # out_enc_level2(2304,96)
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H//2, W//2])  # out_enc_level2(2304,96)
        # out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2, H//2, W//2)  # inp_enc_level3(576,192)
        out_enc_level3 = inp_enc_level3 # out_enc_level3(576,192)
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, [H//4, W//4])  # out_enc_level3(576,192)
        # out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3, H//4, W//4)  # inp_enc_level4(144,384)
        latent = inp_enc_level4  # latent(144,384)
        for layer in self.latent:
            latent = layer(latent, [H//8, W//8])  # latent(144,384)
        # latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent, H//8, W//8)  # inp_dec_level3(144,384)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)  # inp_dec_level3(576,384)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=H//4, w=W//4).contiguous()  # inp_dec_level3(24,24,384)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)  # inp_dec_level3(24,24,192)
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c")  # inp_dec_level3(576,192)
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = inp_dec_level3  # out_dec_level3(576,192)
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, [H//4, W//4])  # out_dec_level3(576,192)
        # out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3, H//4, W//4)  # out_dec_level3(2304,96)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H//2, w=W//2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c")
        # inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, [H//2, W//2])
        # out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2, H//2, W//2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, [H, W])
        # out_dec_level1 = self.decoder_level1(inp_dec_level1)

        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, [H, W])
        # out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=H, w=W)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape  # x(1,48,96,96)
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)  # img_reshape(1,48,1,96,24,4)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)  # (24,384,48)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


###### Dual Gated Feed-Forward Networ
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # x(9216,48)
        x = self.fc1(x)  # x(9216,48*4)
        x = self.act(x)  # x(9216,48*4)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


###### Dual Gated Feed-Forward Networ
class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, act_layer=nn.GELU, drop=0.):
        super(FeedForward, self).__init__()

        self.project_in = nn.Conv2d(in_features, hidden_features*2, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=True)
        self.project_out = nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=True)

    def forward(self, x):  # x(9216,48)
        HW = x.shape[1]
        H = int(HW ** 0.5)
        W = H
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)  # x(1,48,96,96)
        x = self.project_in(x)  # x(192,96,96)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # x1(96,96,96) x2(96,96,96)
        x = F.gelu(x2)*x1 + F.gelu(x1)*x2  # x(96,96,96)
        x = self.project_out(x)  # x(48,96,96)
        x = rearrange(x, "b c h w -> b (h w) c", h=H, w=W)  # x(1,9216,48)
        return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Attention_axial(nn.Module):
    """ Axial Rectangle-Window (axial-Rwin) self-attention with dynamic relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        resolution (int): Input resolution.
        idx (int): The identix of V-Rwin and H-Rwin, -1 is Full Attention, 0 is V-Rwin, 1 is H-Rwin.
        split_size (int): Height or Width of the regular rectangle window, the other is H or W (axial-Rwin).
        dim_out (int | None): The dimension of the attention output, if None dim_out is dim. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    """
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.tmp_H = H_sp
        self.tmp_W = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape  # B=1 N=9216 C=24  # x(1,9216,48)
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)  # x(1,48,96,96)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q,k,v = qkv[0], qkv[1], qkv[2]
        # the side of axial rectangle window changes with input
        if self.resolution != H or self.resolution != W:
            if self.idx == -1:
                H_sp, W_sp = H, W
            elif self.idx == 0:
                H_sp, W_sp = H, self.split_size
            elif self.idx == 1:
                W_sp, H_sp = W, self.split_size
            else:
                print ("ERROR MODE", self.idx)
                exit(0)
            self.H_sp = H_sp
            self.W_sp = W_sp
        else:
            self.H_sp = self.tmp_H
            self.W_sp = self.tmp_W

        B, L, C = q.shape  # L=9216 C=24
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)  # q()
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp, device=attn.device)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp, device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing='ij')) # for pytorch >= 1.10
            # biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w])) # for pytorch < 1.10
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp, device=attn.device)
            coords_w = torch.arange(self.W_sp, device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) # for pytorch >= 1.10
            # coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # for pytorch < 1.10
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)

            pos = self.pos(biases)
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)

        x = x.transpose(1, 2).contiguous().reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x


class CATB_axial(nn.Module):
    """ Axial Cross Aggregation Transformer Block.
    Args:
        dim (int): Number of input channels.
        reso (int): Input resolution.
        num_heads (int): Number of attention heads.
        split_size (int): Height or Width of the axial rectangle window, the other is H or W (axial-Rwin).
        shift_size (int): Shift size for axial-Rwin.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, reso, num_heads,
                 split_size=7, shift_size=0, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.weight_factor = 0.1
        self.window_size = 4

        assert 0 <= self.shift_size < self.split_size, "shift_size must in 0-split_size"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
                    Attention_axial(
                        dim//2, resolution=self.patches_resolution, idx = i,
                        split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
                    for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim) # DW Conv

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        # self.c_attns = CAB(dim, compress_ratio=4, squeeze_factor=16, memory_blocks=256) # compress_ratio=2

    def calculate_mask(self, H, W):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for Rwin
        img_mask_0 = torch.zeros((1, H, self.split_size, 1))
        img_mask_1 = torch.zeros((1, self.split_size, W, 1))
        slices = (slice(-self.split_size, -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for s in slices:
            img_mask_0[:, :, s, :] = cnt
            img_mask_1[:, s, :, :] = cnt
            cnt += 1

        # calculate mask for V-Shift
        img_mask_0 = img_mask_0.view(1, H // H, H, self.split_size // self.split_size, self.split_size, 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, self.split_size, 1)
        mask_windows_0 = img_mask_0.view(-1, H * self.split_size)
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))
        num_v = W // self.split_size
        attn_mask_0_la = torch.zeros((num_v,H * self.split_size,H * self.split_size))
        attn_mask_0_la[-1] = attn_mask_0

        # calculate mask for H-Shift
        img_mask_1 = img_mask_1.view(1, self.split_size // self.split_size, self.split_size, W // W, W, 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size, W, 1)
        mask_windows_1 = img_mask_1.view(-1, self.split_size * W)
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))
        num_h = H // self.split_size
        attn_mask_1_la = torch.zeros((num_h,W * self.split_size,W * self.split_size))
        attn_mask_1_la[-1] = attn_mask_1

        return attn_mask_0_la, attn_mask_1_la

    def forward(self, x, x_size):  # x(1,9216,48) x_size[96,96]
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H , W = x_size  # H=96 W=96
        B, L, C = x.shape  # L=9216 C=48
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)  # img(1,9216,48)
        qkv = self.qkv(img)  # qkv(1,9216,144)
        qkv = qkv.reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C qkv(3,1,9216,48)
        # v without partition
        v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)

        if self.shift_size > 0:
            qkv = qkv.view(3, B, H, W, C)
            # V-Shift
            qkv_0 = torch.roll(qkv[:, :, :, :, :C // 2], shifts=-self.shift_size, dims=3)
            qkv_0 = qkv_0.view(3, B, L, C // 2)
            # H-Shift
            qkv_1 = torch.roll(qkv[:, :, :, :, C // 2:], shifts=-self.shift_size, dims=2)
            qkv_1 = qkv_1.view(3, B, L, C // 2)

            if self.patches_resolution != H or self.patches_resolution != W:
                mask_tmp = self.calculate_mask(H, W)
                # V-Rwin
                x1_shift = self.attns[0](qkv_0, H, W, mask=mask_tmp[0].to(x.device))
                # H-Rwin
                x2_shift = self.attns[1](qkv_1, H, W, mask=mask_tmp[1].to(x.device))

            else:
                # V-Rwin
                x1_shift = self.attns[0](qkv_0, H, W, mask=self.attn_mask_0)
                # H-Rwin
                x2_shift = self.attns[1](qkv_1, H, W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=self.shift_size, dims=2)
            x2 = torch.roll(x2_shift, shifts=self.shift_size, dims=1)
            x1 = x1.view(B, L, C // 2).contiguous()
            x2 = x2.view(B, L, C // 2).contiguous()
            # Concat
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            # V-Rwin
            x1 = self.attns[0](qkv[:, :, :, :C // 2], H, W).view(B, L, C // 2).contiguous()
            # H-Rwin
            x2 = self.attns[1](qkv[:, :, :, C // 2:], H, W).view(B, L, C // 2).contiguous()
            # Concat
            attened_x = torch.cat([x1, x2], dim=2)

        # Locality Complementary Module
        lcm = self.get_v(v)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        attened_x = attened_x + lcm  # attened_x(1,9216,48)

        ## 光谱分流模块
        # attened_x = rearrange(attened_x, 'b n (g d) -> b n ( d g)', g=4)  # attened_x(1,9216,48)



        # ## 光谱增强模块
        # attened_x = attened_x.view(B, H, W, C)  # x(1,96,96,48)
        #
        # # cyclic shift 循环移位操作
        # if self.shift_size > 0:
        #     shifted_x = torch.roll(attened_x, shifts=(-self.window_size//2, -self.window_size//2), dims=(1, 2))
        # else:
        #     shifted_x = attened_x  # shifted_x(1,96,96,48)
        #
        #
        #
        # # partition windows  窗口分区
        # x_windows = window_partition(shifted_x, self.window_size)  # x_windows(64,12,12,48)
        # x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # x_windows(64,144,48)  144个块，每个块8*8  通道为48
        #
        # attn_windows = self.c_attns(x_windows)  # attn_windows(64,144,48)
        #
        # # merge windows  合并窗口
        # attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # attn_windows(144,8,8,48)
        # shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C  # shifted_x(1,96,96,48)
        #
        # # reverse cyclic shift  反向循环移位
        # if self.shift_size > 0:
        #     attened_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        # else:
        #     attened_x = shifted_x  # x(1,96,96,48)
        #
        # attened_x = attened_x.view(B, H * W, C)  # x(1,9216,48)
        # # attened_c = self.c_attns(img, H, W)  # x(1,9216,48)
        # # attened_x = attened_x + 0.05*attened_c



        attened_x = self.proj(attened_x)
        attened_x = self.proj_drop(attened_x)

        x = x + self.drop_path(attened_x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=10, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.proj(x)  # x(1,48,96,96)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()  # x(1,48,9216)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, H, W):  # 通道数/2 因为长宽各除2，所以除4
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=4, squeeze_factor=30, memory_blocks=128):  # num_feat=48
        super(CAB, self).__init__()
        self.num_feat = num_feat
        self.cab = nn.Sequential(
            nn.Linear(num_feat, num_feat // compress_ratio),
            nn.GELU(),
            nn.Linear(num_feat // compress_ratio, num_feat))

        self.ca = ChannelAttention(num_feat, squeeze_factor, memory_blocks)

    def forward(self, x):  # x(1,9216,48) (1,2304,96) (1,576,192) (1,144,384)
        x = self.cab(x)  # x(1,9216,48)
        x = self.ca(x)  # x(1,9216,48)
        return x  # x(144,64,48)

    def flops(self, shape):
        flops = 0
        H, W = shape
        flops += self.num_feat * H * W
        return flops


#### Cross-layer Attention Fusion Block  跨层注意力融合块
class LAM_Module_v2(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim,bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self, x):  # x(1,3,48,96,96)
        """
            inputs :
                x : input feature maps( B  N  C  H  W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()  # x(1,3,48,96,96)

        x_input = x.reshape(m_batchsize,N*C, height, width)  # x_input(1,144,96,96)
        qkv = self.qkv_dwconv(self.qkv(x_input))  # qkv(1,432,96,96)
        q, k, v = qkv.chunk(3, dim=1)  # q(1,144,96,96)
        q = q.view(m_batchsize, N, -1)  # q(1,3,442368)
        k = k.view(m_batchsize, N, -1)  # k(1,3,442368)
        v = v.view(m_batchsize, N, -1)  # v(1,3,442368)

        q = torch.nn.functional.normalize(q, dim=-1)  # q(1,3,442368)
        k = torch.nn.functional.normalize(k, dim=-1)  # k(1,3,442368)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # attn(1,3,3)
        attn = attn.softmax(dim=-1)  # attn(1,3,3)

        out_1 = (attn @ v)  # out_1(1,3,442368)
        out_1 = out_1.view(m_batchsize, -1, height, width)  # out_1(1,144,96,96)

        out_1 = self.project_out(out_1)  # out_1(1,144,96,96)
        out_1 = out_1.view(m_batchsize, N, C, height, width)  # out_1(1,3,48,96,96)

        out = out_1+x  # out(1,3,48,96,96)
        out = out.view(m_batchsize, -1, height, width)  # out_1(1,144,96,96)
        return out


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16, memory_blocks=128):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.subnet = nn.Sequential(

            nn.Linear(num_feat, num_feat // squeeze_factor),
            # nn.ReLU(inplace=True)
        )
        self.upnet = nn.Sequential(
            nn.Linear(num_feat // squeeze_factor, num_feat),
            # nn.Linear(num_feat, num_feat),
            nn.Sigmoid())
        self.mb = torch.nn.Parameter(torch.randn(num_feat // squeeze_factor, memory_blocks))
        self.low_dim = num_feat // squeeze_factor

    def forward(self, x):  # x(1,9216,48)
        b, n, c = x.shape  # x(144,64,48)
        t = x.transpose(1, 2)  # t(144,48,64)
        y = self.pool(t).squeeze(-1)  # y(144,48) 池化了

        low_rank_f = self.subnet(y).unsqueeze(2)  # low_rank_f(144,3,1)

        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)  # mbg(144,3,256)
        f1 = (low_rank_f.transpose(1, 2)) @ mbg  # f1(144,1,256)
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)  # f_dic_c(144,1,256)  # get the similarity information  获取相似性信息
        y1 = f_dic_c @ mbg.transpose(1, 2)
        y2 = self.upnet(y1)
        out = x * y2
        return out


def window_partition(x, window_size):  # x(1,96,96,48) window_size=8
    B, H, W, C = x.shape  # x(1,96,96,48)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)  # x(1,12,8,12,8,48)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)  # windows(144,8,8,48)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class Conv2D_pxp(nn.Module):

    def __init__(self, in_ch, out_ch, k, s, p):
        super(Conv2D_pxp, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.PReLU(out_ch)

    def forward(self, input):
        return self.relu(self.bn(self.conv(input)))


class CC_Module(nn.Module):

    def __init__(self):
        super(CC_Module, self).__init__()


        self.layer1_1 = Conv2D_pxp(16, 32, 3, 1, 1)
        self.layer1_2 = Conv2D_pxp(16, 32, 5, 1, 2)
        self.layer1_3 = Conv2D_pxp(16, 32, 7, 1, 3)

        self.layer2_1 = Conv2D_pxp(96, 32, 3, 1, 1)
        self.layer2_2 = Conv2D_pxp(96, 32, 5, 1, 2)
        self.layer2_3 = Conv2D_pxp(96, 32, 7, 1, 3)

        self.local_attn_r = CBAM(64)
        self.local_attn_g = CBAM(64)
        self.local_attn_b = CBAM(64)

        self.layer3_1 = Conv2D_pxp(192, 16, 3, 1, 1)
        self.layer3_2 = Conv2D_pxp(192, 16, 5, 1, 2)
        self.layer3_3 = Conv2D_pxp(192, 16, 7, 1, 3)

        self.d_conv1 = nn.ConvTranspose2d(in_channels=48, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.d_bn1 = nn.BatchNorm2d(num_features=96)
        self.d_relu1 = nn.PReLU(96)

        self.global_attn_rgb = CBAM(96)

        self.d_conv2 = nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.d_bn2 = nn.BatchNorm2d(num_features=48)
        self.d_relu2 = nn.PReLU(48)

    def forward(self, input):  # input(1,48,96,96)
        input_1 = input[:, 0:16, :, :]  # input_1(1,16,96,96)
        input_2 = input[:, 16:32, :, :]
        input_3 = input[:, 32:48, :, :]

        # layer 1
        l1_1 = self.layer1_1(input_1)  # input(1,32,96,96)
        l1_2 = self.layer1_2(input_2)
        l1_3 = self.layer1_3(input_3)

        # Input to layer 2
        input_l2 = torch.cat((l1_1, l1_2), 1)
        input_l2 = torch.cat((input_l2, l1_3), 1)  # input_l2(1,96,96,96)  # input_l2(1,96,96,96)

        # layer 2
        l2_1 = self.layer2_1(input_l2)  # l2_1(1,32,96,96)
        l2_1 = self.local_attn_r(torch.cat((l2_1, l1_1),1))  # l2_1(1,64,96,96)

        l2_2=self.layer2_2(input_l2)
        l2_2 = self.local_attn_g(torch.cat((l2_2, l1_2),1))

        l2_3=self.layer2_3(input_l2)
        l2_3 = self.local_attn_b(torch.cat((l2_3, l1_3),1))

        # Input to layer 3
        input_l3 = torch.cat((l2_1, l2_2), 1)
        input_l3 = torch.cat((input_l3, l2_3), 1)

        # layer 3
        l3_1 = self.layer3_1(input_l3)
        l3_2 = self.layer3_2(input_l3)
        l3_3 = self.layer3_3(input_l3) # 32

        # input to decoder unit
        temp_d1 = torch.add(input_1, l3_1) # 16
        temp_d2 = torch.add(input_2, l3_2)
        temp_d3 = torch.add(input_3, l3_3)

        input_d1 = torch.cat((temp_d1, temp_d2), 1) # 32
        input_d1 = torch.cat((input_d1, temp_d3), 1)  # 48



        # decoder
        output_d1 = self.d_relu1(self.d_bn1(self.d_conv1(input_d1)))
        output_d1 = self.global_attn_rgb(output_d1)
        final_output = self.d_relu2(self.d_bn2(self.d_conv2(output_d1)))

        return final_output