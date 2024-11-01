import time
import datetime


import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import Pvt_v2_b2
import os


import functools
from cc_attention import CrissCrossAttention


from inplace_abn import InPlaceABN, InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')

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
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4

        self.conva = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            InPlaceABNSync(inter_channels)
        )
        self.cca = CrissCrossAttention(inter_channels)

        self.convb = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            InPlaceABNSync(inter_channels)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=2):
        output = self.conva(x)
        print("%%%%%%%%%%%%%%%%%%%")
        print(output.size())
        for i in range(recurrence):
            output = self.cca(output)
        # print("*********************")
        # print(output.size())
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output


class PolypAttenionPVT(nn.Module):
    def __init__(self, channel=32):
        super(PolypAttenionPVT, self).__init__()
        #--------------------------------------#
        #   获取两个特征层
        #   浅层特征    [32,64,64]
        #   深层特征    [256,64,64]
        # --------------------------------------#
        self.backbone = Pvt_v2_b2()
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.RFB = RFB_modified(512, channel)
        self.RCCAttenion = RCCAModule(64, channel, num_classes=2)

        self.UP = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.out_RFB = nn.Sequential(
            nn.Conv2d(channel, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.out_RCCA = nn.Sequential(
            nn.Conv2d(channel, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(32+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, 1, 1)
        # self.out_conv = nn.Conv2d(channel, 1, 3)
        self.out_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        #print('&&&&&&&&&&&&')
        #print(x.size())
        x1 = pvt[0]
        # print("#############")
        # print(x1.size())
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x1_1 = self.RCCAttenion(x1)
        # print("@@@@@@@@@@@@@@")
        # print(x1_1.size())
        x1_2 = self.out_RCCA(x1_1)

        x2_1 = self.RFB(x4)
        x2_2 = self.out_RFB(x2_1)
        x2_3 = self.UP(x2_2)
        x_cat = torch.cat((x1_2, x2_3), 1)
        x2_4 = self.out_conv(x_cat)
        x = self.cls_conv(x2_4)
        x_out = self.out_up(x)
        # print("^^^^^^^^^^^^^")
        # print(x_out.size())
        return x_out


if __name__ == '__main__':
    model = PolypAttenionPVT().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    x_out = model(input_tensor)
    print(x_out.size())