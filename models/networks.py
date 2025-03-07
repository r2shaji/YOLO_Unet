""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, scale_factor=2, bilinear=False):
        """
        Args:
            in_channels: Number of channels in the input (from previous layer).
            skip_channels: Number of channels in the skip connection.
            out_channels: Number of channels to output after the double conv.
            scale_factor: Factor by which to upsample spatially. For layers with matching spatial
                          dimensions (no upsampling needed) use scale_factor=1.
            bilinear: If True, use bilinear upsampling (which preserves channels); otherwise,
                      use ConvTranspose2d to map in_channels to skip_channels.
        """
        super(Up, self).__init__()
        self.scale_factor = scale_factor
        
        if scale_factor > 1:
            if bilinear:
                self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
                up_channels = in_channels
            else:
                self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=scale_factor, stride=scale_factor)
                up_channels = skip_channels
        else:
            # perform a convolutional refinement
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            up_channels = in_channels
        
        # After upsampling, concatenate with the skip connection.
        self.conv = DoubleConv(up_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        print(x1.shape,x2.shape)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        print("x.shape",x.shape)
        return self.conv(x)
    

# UpNoSkip block for initial upsampling when no skip is fused.
class UpNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, bilinear=False):
        super(UpNoSkip, self).__init__()
        self.scale_factor = scale_factor
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)
        self.conv = DoubleConv(out_channels, out_channels)
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)