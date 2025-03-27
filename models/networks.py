import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

class DoubleConv(nn.Module):
    
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

    def __init__(self, in_channels, skip_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

        if scale_factor > 1:
            conv_out_channels = skip_channels * (scale_factor ** 2)
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, conv_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(conv_out_channels),
                nn.ReLU(inplace=True),            
                nn.PixelShuffle(scale_factor)
            )
            conv_in_channels = skip_channels + skip_channels
        else:
            self.up = nn.Sequential(
                 nn.Conv2d(in_channels, skip_channels, kernel_size=3, padding=1, bias=False),
                 nn.BatchNorm2d(skip_channels),
                 nn.ReLU(inplace=True)
            )
            conv_in_channels = skip_channels + skip_channels

        self.conv = DoubleConv(conv_in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1) 

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX > 0 or diffY > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        if x1.size()[2] > x2.size()[2] or x1.size()[3] > x2.size()[3]:
             x1 = x1[:, :, :x2.size()[2], :x2.size()[3]]

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpNoSkip(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

        if scale_factor > 1:
            conv_out_channels = out_channels * (scale_factor ** 2)
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, conv_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(conv_out_channels),
                nn.ReLU(inplace=True),            
                nn.PixelShuffle(scale_factor)
            )
        else: 
            self.up = nn.Sequential(
                 nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(inplace=True)
             )

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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)
    
class PerceptualLoss(nn.Module):

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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