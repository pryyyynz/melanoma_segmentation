import torch
import torch.nn as nn


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class GatedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GatedConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x_conv = self.conv(x)
        x_mask = torch.sigmoid(self.mask_conv(x))
        return x_conv * x_mask


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out += residual
        out = self.relu(out)
        return out


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
        self.r1 = ResidualBlock(out_channels, out_channels, stride)
        self.r2 = ResidualBlock(out_channels, out_channels, stride)
        self.r3 = ResidualBlock(out_channels, out_channels, stride)

    def forward(self, x0, x1, x2, x3, x4):
        # Pass input through f1, f2, and g1
        x_f1 = self.f1(x0)
        x_f2 = self.f2(x1)
        x_g1 = self.g1(x_f1 + x_f2)

        # Pass output of g1 through r1
        x_r1 = self.r1(x_g1)

        # Pass input through f3 and pass it along with r1 output through g2
        x_f3 = self.f3(x2)
        x_g2 = self.g2(x_f3 + x_r1)

        # Pass output of g2 through r2
        x_r2 = self.r2(x_g2)

        # Pass input through f4 and pass it along with r2 output through g3
        x_f4 = self.f4(x3)
        x_g3 = self.g3(x_f4 + x_r2)

        # Pass output of g3 through r3
        x_r3 = self.r3(x_g3)

        # Pass input through f5 and pass it along with r3 output through g4
        x_f5 = self.f5(x4)
        x_g4 = self.g4(x_f5 + x_r3)

        return x_g4
