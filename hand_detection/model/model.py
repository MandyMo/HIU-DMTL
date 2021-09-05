

import sys
from os import path as osp
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(cur_dir, '..'))

import torch
from torch.hub import MODULE_HUBCONF
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.module import Module

def get_backbone(s_backbone='resnet18', pretrained=True, *args, **kargs):
    if s_backbone=='resnet18':
        try:
            from .resnet import resnet18
        except:
            from resnet import resnet18
        return resnet18(pretrained=pretrained), [512,256,128,64]
    else:
        raise 'backbone is not supported currently...'

class Model(nn.Module):
    def __init__(self, s_backbone='resnet18', c_out=5, deconv_outs = [32, 16, 8], *args, **kargs):
        super(Model, self).__init__()
        self.backbone, self.n_chs = get_backbone(s_backbone=s_backbone)

        #now define the heads
        self.up1 = nn.Sequential(
            nn.Conv2d(self.n_chs[0], deconv_outs[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(deconv_outs[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(deconv_outs[0], deconv_outs[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(deconv_outs[0]),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(self.n_chs[1]+deconv_outs[0], deconv_outs[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(deconv_outs[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(deconv_outs[1], deconv_outs[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(deconv_outs[1]),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(self.n_chs[2]+deconv_outs[1], deconv_outs[2], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(deconv_outs[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(deconv_outs[2], deconv_outs[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(deconv_outs[2]),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(self.n_chs[3]+deconv_outs[2], deconv_outs[2], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(deconv_outs[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(deconv_outs[2], c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, X, *args, **kargs):
        c2, c3, c4, c5 = self.backbone(X)
        up1 = self.up1(c5)
        up2 = self.up2(torch.cat([up1, c4], 1))
        up3 = self.up3(torch.cat([up2, c3], 1))
        out = self.out(torch.cat([up3, c2], 1))

        return out

if __name__ == '__main__':
    model = Model().cuda().eval().float()
    x = torch.randn(1, 3, 256, 256).float().cuda()
    print(model(x).shape)