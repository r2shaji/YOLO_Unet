""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


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
    
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        self.device = torch.device("cpu")  
        
        with torch.no_grad():
            self.vgg_relu_3_3 = self.contentFunc(15).to(self.device)
            self.vgg_relu_2_2 = self.contentFunc(8).to(self.device)
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def contentFunc(self, relu_layer):
        cnn = models.vgg19(pretrained=True).features
        model = nn.Sequential()

        cnn = cnn.to(self.device)
        model = model.to(self.device)
        model.eval()

        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == relu_layer:
                break

        return model

    def get_loss(self, fakeIm, realIm):

        fakeIm = self.transform(fakeIm)
        realIm = self.transform(realIm)

        f_fake_2_2 = self.vgg_relu_2_2(fakeIm)
        f_real_2_2 = self.vgg_relu_2_2(realIm)

        f_fake_3_3 = self.vgg_relu_3_3(fakeIm)
        f_real_3_3 = self.vgg_relu_3_3(realIm)

        f_real_2_2_no_grad = f_real_2_2.detach()
        f_real_3_3_no_grad = f_real_3_3.detach()

        mse = nn.MSELoss()
        loss = mse(f_fake_2_2, f_real_2_2_no_grad) + mse(f_fake_3_3, f_real_3_3_no_grad)

        return loss / 2.0

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)