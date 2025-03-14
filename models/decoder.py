import torch
import torch.nn as nn
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import math
from PIL import Image
from models.networks import Up, OutConv, UpNoSkip
import util

class YOLO_UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out=3, bilinear=False):
        super(YOLO_UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        self.up1 = UpNoSkip(in_channels=384, out_channels=256, scale_factor=2, bilinear=bilinear)
        self.up2 = Up(in_channels=256, skip_channels=192, out_channels=256, scale_factor=1, bilinear=bilinear)
        self.up3 = Up(in_channels=256, skip_channels=192, out_channels=256, scale_factor=1, bilinear=bilinear)
        self.up4 = Up(in_channels=256, skip_channels=96, out_channels=128, scale_factor=2, bilinear=bilinear)
        self.up5 = Up(in_channels=128, skip_channels=96, out_channels=128, scale_factor=1, bilinear=bilinear)
        self.up6 = Up(in_channels=128, skip_channels=3, out_channels=64, scale_factor=4, bilinear=bilinear)
        self.outc = OutConv(64, n_channels_out)

    def forward(self, yolo_feats):
        x0 = yolo_feats[0].clone()  # [B,   3, 512, 512]
        x1 = yolo_feats[1].clone()  # [B,  96, 128, 128]
        x2 = yolo_feats[2].clone()  # [B,  96, 128, 128]
        x3 = yolo_feats[3].clone()  # [B, 192,  64,  64]
        x4 = yolo_feats[4].clone()  # [B, 192,  64,  64]
        x5 = yolo_feats[5].clone()  # [B, 384,  32,  32]

        f1 = self.up1(x5)         # f1: [B,256,64,64]
        f2 = self.up2(f1, x4)       # f2: [B,256,64,64]
        f3 = self.up3(f2, x3)       # f3: [B,256,64,64]
        f4 = self.up4(f3, x2)       # f4: [B,128,128,128]
        f5 = self.up5(f4, x1)       # f5: [B,128,128,128]
        f6 = self.up6(f5, x0)       # f6: [B,64,512,512]
        output = self.outc(f6)
        return output

    def use_checkpointing(self):
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.up5 = torch.utils.checkpoint(self.up5)
        self.up6 = torch.utils.checkpoint(self.up6)
        self.outc = torch.utils.checkpoint(self.outc)