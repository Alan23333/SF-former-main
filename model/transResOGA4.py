import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
#中间只加残差通道注意力
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class TransformerEBlock(nn.Module):
    def __init__(self, in_c, out_c, depth):
        super().__init__()
        self.Embedding = nn.Sequential(
            nn.Linear(in_c, out_c)
        )

        self.transformer = nn.ModuleList([CATB_axial(dim=out_c,
                                                           num_heads=2,
                                                           reso=24,
                                                           mlp_ratio=4,
                                                           qkv_bias=True,
                                                           qk_scale=None,
                                                           split_size=2,
                                                           drop=0.,
                                                           attn_drop=0.,
                                                           drop_path=0.,
                                                           act_layer=nn.GELU,
                                                           norm_layer=nn.LayerNorm,
                                                           shift_size=0 if (i % 2 == 0) else 1,
                                                           )
                                            for i in range(depth)])



    def forward(self,X):
        _, _, H, W = X.shape  # X(16,64,24,24)
        E = rearrange(X, 'B c H W -> B (H W) c', H=H, W=W)  # E(16,576,64)
        E = self.Embedding(E)  # E(16,578,80)
        y1 = E
        for layer in self.transformer:
            y1 = layer(y1, [H, W])  # y1(16,578,80)
        out = torch.add(E, y1)
        out = rearrange(out, 'B (H W) C -> B C H W', H=H, W=W)
        return out



class DownTrans(nn.Module):
    def __init__(self,in_c,out_c,depth): #in_c输入通道，out_c输出通道，前面赋值了
        super().__init__()

        self.avgPool = nn.AvgPool2d(2)
        self.TransformerEBlock = TransformerEBlock(in_c, out_c,depth)

    def forward(self,X): #x=(16,64,48,48)
        X = self.avgPool(X)  # X(16,64,24,24)
        out = self.TransformerEBlock(X) #out=(16,80,24,24) net就是图3的A过程
        return out

class UpTrans(nn.Module):
    def __init__(self,in_c, mid_c, out_c, depth):
        super().__init__()
        self.trans = nn.Sequential(
            TransformerEBlock(in_c + mid_c, out_c, depth),
            nn.ConvTranspose2d(in_channels=out_c, out_channels=out_c, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU()
        )

    def forward(self,X,Y):
        x2 = torch.cat((X,Y),dim=1)
        out = self.trans(x2)
        return out


class GSNet(nn.Module):
    def __init__(self,hs_band,ms_band):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=hs_band+ms_band,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(),
        )

        self.conv = nn.Conv2d(in_channels=128+64,out_channels=hs_band,kernel_size=1,stride=1,padding=0)

        self.down1 = DownTrans(64,80,2)
        self.down2 = DownTrans(80,96,2)
        self.down3 = DownTrans(96,112,2)

        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels=112, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=2,stride=2,padding=0),
        )
        self.up2 = UpTrans(128,96,128,2)
        self.up3 = UpTrans(128,80,128,2)

        self.skips1 = HAB(64)
        self.skips2 = HAB(80)
        self.skips3 = HAB(96)

    def forward(self, y,z): #y=(16,31,6,6) z=(16,1,48,48)
        y_up = F.interpolate(y,scale_factor=8,mode='bicubic',align_corners=False) #(16,31,48,48) 张量y进行上采样，通过使用双三次插值将其调整大小为原来的8倍（6乘以8=48）
        y1 = torch.cat((y_up,z),dim=1) #y1=(16,32,48,48) cat函数连接张量
        y1 = self.conv1(y1)#y1=(16,64,48,48) 3乘3卷积改变通道数
        y2 = self.down1(y1) #y2=(16,80,24,24)
        y3 = self.down2(y2) #y3=(16,96,12,12)
        y4 = self.down3(y3)  #y4=(16,122,6,6)
        y4 = self.up1(y4)    #y4=(16,128,12,12)
        yo1 = self.skips1(y1)
        yo2 = self.skips2(y2)
        yo3 = self.skips3(y3)
        out1 = self.up2(y4, yo3) #out1=(16,128,24,24)
        out2 = self.up3(out1, yo2) #out2=(16,128,48,48)
        out3 = torch.cat((out2, yo1),dim=1) #out3=(16,192,48,48)
        out = self.conv(out3) #out3=(16,31,48,48)
        return out


class HAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=4, squeeze_factor=16):
        super(HAB, self).__init__()
        self.num_feat = num_feat
        self.ab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
        )
        self.ca = ChannelAttention(num_feat//2, squeeze_factor)

        self.fa = FilterAttention(num_feat//2, num_feat//2)

    def forward(self, x):
        xa = self.ab(x)
        ca = self.ca(xa[:, :self.num_feat//2, :, :])
        fa = self.fa(xa[:, self.num_feat//2:, :, :])

        xx = torch.cat((ca, fa), dim=1)
        out = x + xx
        return out


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.a = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        xx = self.a(x)
        y = self.attention(xx)
        return x * y



class FilterAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FilterAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature_map = self.conv(x)
        attention = self.sigmoid(feature_map)
        output = feature_map * attention
        return output


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

    def forward(self, x, x_size):  # x(16,578,80)
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H , W = x_size
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C
        # v without partition
        v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)

        if self.shift_size > 0:
            qkv = qkv.view(3, B, H, W, C)
            # V-Shift
            qkv_0 = torch.roll(qkv[:,:,:,:,:C//2], shifts=-self.shift_size, dims=3)
            qkv_0 = qkv_0.view(3, B, L, C//2)
            # H-Shift
            qkv_1 = torch.roll(qkv[:,:,:,:,C//2:], shifts=-self.shift_size, dims=2)
            qkv_1 = qkv_1.view(3, B, L, C//2)

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
            x1 = x1.view(B, L, C//2).contiguous()
            x2 = x2.view(B, L, C//2).contiguous()
            # Concat
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            # V-Rwin
            x1 = self.attns[0](qkv[:,:,:,:C//2], H, W).view(B, L, C//2).contiguous()
            # H-Rwin
            x2 = self.attns[1](qkv[:,:,:,C//2:], H, W).view(B, L, C//2).contiguous()
            # Concat
            attened_x = torch.cat([x1,x2], dim=2)

        # Locality Complementary Module
        lcm = self.get_v(v)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        attened_x = attened_x + lcm

        attened_x = self.proj(attened_x)
        attened_x = self.proj_drop(attened_x)

        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


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
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
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

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
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


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
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