import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import Pvt_v2_b2
import os

import functools
from cc_attention import CrissCrossAttention

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4

        self.conva = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.cca = CrissCrossAttention(inter_channels)

        self.convb = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, recurrence=2):
        output = self.conva(x)
        # print("%%%%%%%%%%%%%%%%%%%")
        # print(output.size())
        for i in range(recurrence):
            output = self.cca(output)
        # print("*********************")
        # print(output.size())
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output


class PolypFormer(nn.Module):
    def __init__(self, channel=32):
        super(PolypFormer, self).__init__()
        #--------------------------------------#
        #   获取两个特征层
        #   浅层特征    [64,88,88]
        #   深层特征    [128,44,44]、 [320,22,22]、 [512,11,11]
        # --------------------------------------#
        self.backbone = Pvt_v2_b2()
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.RFB_4 = RFB_modified(512, channel)
        self.RFB_3 = RFB_modified(320, channel)
        self.RFB_2 = RFB_modified(128, channel)
        self.RCCAttention = RCCAModule(64, channel)

        self.UP = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.cov_RFB = nn.Sequential(
            nn.Conv2d(channel, 64, 1),
            nn.ReLU(inplace=True)
        )
        self.out_RCCA = nn.Sequential(
            nn.Conv2d(channel, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_up = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(64+64, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, 1, 1)
        self.out_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x1_1 = self.RCCAttention(x1)
        x1_2 = self.out_RCCA(x1_1)

        x4_1 = self.RFB_4(x4)                   # 32, 11, 11
        x4_2 = self.cov_RFB(x4_1)               # 64, 11, 11
        x4_3 = self.UP(x4_2)                    # 64, 22, 22

        x3_1 = self.RFB_3(x3)                   # 32, 22, 22
        x3_2 = self.cov_RFB(x3_1)               # 64, 22, 22

        x_cat1 = torch.cat((x4_3, x3_2), 1)      # 128,22, 22
        x_out1 = self.conv_up(x_cat1)            # 64, 22, 22
        x3_3 = self.UP(x_out1)                   # 64, 44, 44

        x2_1 = self.RFB_2(x2)                   # 32, 44, 44
        x2_2 = self.cov_RFB(x2_1)               # 64, 44, 44
        x_cat2 = torch.cat((x3_3, x2_2), 1)     # 128, 44, 44
        x_out2 = self.conv_up(x_cat2)           # 64, 44, 44
        x2_3 = self.UP(x_out2)                  # 64, 88, 88

        x_cat = torch.cat((x2_3, x1_2), 1)      # 96,88,88
        # x_out3 = self.conv_up(x_cat)          # 64, 88, 88

        x2_4 = self.out_conv(x_cat)
        x = self.cls_conv(x2_4)
        x_out = self.out_up(x)
        # print("^^^^^^^^^^^^^")
        # print(x_out.size())
        return x_out


if __name__ == '__main__':
    model = PolypFormer().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    x_out = model(input_tensor)
    print(x_out.size())