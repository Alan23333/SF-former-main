import numpy as np
import torch
import torch.nn as nn
from model import common
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F


def make_model(opt):
    return XXX(opt)


class XXX(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(XXX, self).__init__()
        self.scale = opt.scale
        self.phase = int(np.log2(opt.scale))
        n_blocks = opt.n_blocks
        n_feats = opt.n_feats
        kernel_size = 3

        act = nn.ReLU(True)

        self.head = conv(opt.n_colors, n_feats, kernel_size)

        self.down = [
            common.DownBlock(opt, 2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
            ) for p in range(self.phase)
        ]

        self.down = nn.ModuleList(self.down)

        up_body_blocks = [[
            common.RCAB(
                conv, n_feats * pow(2, p), kernel_size, act=act
            ) for _ in range(n_blocks)
        ] for p in range(self.phase, 1, -1)
        ]

        up_body_blocks.insert(0, [
            common.RCAB(
                conv, n_feats * pow(2, self.phase), kernel_size, act=act
            ) for _ in range(n_blocks)
        ])

        # The fisrt upsample block
        up = [[
            common.Upsampler(conv, 2, n_feats * pow(2, self.phase), act=False),
            conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                common.Upsampler(conv, 2, 2 * n_feats * pow(2, p), act=False),
                conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # tail conv that output sr imgs
        self.tail = conv(n_feats * 2, opt.n_colors, kernel_size)

    def draw_features(self, width, height, x, savename):
        # tic=time.time()
        fig = plt.figure(figsize=(60, 60))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        for i in range(width * height):
            plt.subplot(height, width, i + 1)
            plt.axis('off')
            img = x[0, i, :, :]
            pmin = np.min(img)
            pmax = np.max(img)
            img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255   # float在[0，1]之间，转换成0-255
            img = img.astype(np.uint8)  # 转成unit8
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
            img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
            plt.imshow(img)
        fig.savefig(savename, dpi=100)
        fig.clf()
        plt.close()
        # print("time:{}".format(time.time()-tic))

    def forward(self, x):
        # preprocess
        x = self.head(x)

        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        # savepath = "."
        # self.draw_features(4, 8, (F.interpolate(copies[2], scale_factor=8, mode='bilinear')).cpu().detach().numpy(),
        #               "{}/Out_Merge_8_img3.png".format(savepath))

        # up phases
        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks[idx](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)
            # output sr imgs

            # savepath = "."
            # self.draw_features(4, 8, (F.interpolate(x, scale_factor=8, mode='bilinear')).cpu().detach().numpy(),
            #                    "{}/Out_Merge_8_img4.png".format(savepath))
        sr = self.tail(x)

        return sr