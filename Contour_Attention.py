import torch
import torch.nn as nn
import torch.nn.functional as F


# 1x1 Convolution layer
class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# Gated Convolution layer
class GatedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GatedConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x_conv = self.conv(x)
        x_mask = torch.sigmoid(self.mask_conv(x))
        return x_conv * x_mask


# Residual Block with dilated convolutions
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


# Contour Attention module
class ContourAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ContourAttention, self).__init__()
        # Define 1x1 convolutions
        self.f1 = Conv1x1(in_channels, out_channels)
        self.f2 = Conv1x1(in_channels, out_channels)
        self.f3 = Conv1x1(in_channels, out_channels)
        self.f4 = Conv1x1(in_channels, out_channels)
        self.f5 = Conv1x1(in_channels, out_channels)

        # Define gated convolutions
        self.g1 = GatedConvolution(out_channels, out_channels, kernel_size, stride, padding)
        self.g2 = GatedConvolution(out_channels, out_channels, kernel_size, stride, padding)
        self.g3 = GatedConvolution(out_channels, out_channels, kernel_size, stride, padding)
        self.g4 = GatedConvolution(out_channels, out_channels, kernel_size, stride, padding)

        # Define residual convolutions
        self.r1 = ResidualBlock(out_channels, out_channels)
        self.r2 = ResidualBlock(out_channels, out_channels)
        self.r3 = ResidualBlock(out_channels, out_channels)

    def forward(self, x0, x1, x2, x3, x4):
        # Interpolate all inputs to size (128, 128)
        x0 = F.interpolate(x0, size=(128, 128))
        x1 = F.interpolate(x1, size=(128, 128))
        x2 = F.interpolate(x2, size=(128, 128))
        x3 = F.interpolate(x3, size=(128, 128))
        x4 = F.interpolate(x4, size=(128, 128))

        # Process x0 through f1 and x1 through f2
        x_f1 = self.f1(x0)
        x_f2 = self.f2(x1)

        # Combine outputs of f1 and f2, process through g1
        x_g1 = self.g1(x_f1 + x_f2)

        # Process output of g1 through r1
        x_r1 = self.r1(x_g1)
        x_r1 = F.interpolate(x_r1, size=(128, 128))

        # Process x2 through f3, combine with output of r1, and process through g2
        x_f3 = self.f3(x2)
        x_g2 = self.g2(x_f3 + x_r1)

        # Process output of g2 through r2
        x_r2 = self.r2(x_g2)
        x_r2 = F.interpolate(x_r2, size=(128, 128))

        # Process x3 through f4, combine with output of r2, and process through g3
        x_f4 = self.f4(x3)
        x_g3 = self.g3(x_f4 + x_r2)

        # Process output of g3 through r3
        x_r3 = self.r3(x_g3)
        x_r3 = F.interpolate(x_r3, size=(128, 128))

        # Process x4 through f5, combine with output of r3, and process through g4
        x_f5 = self.f5(x4)
        x_g4 = self.g4(x_f5 + x_r3)

        return x_g4
