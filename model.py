from Contour_Attention import ContourAttention, ResidualBlock
import torch
import torch.nn as nn
import torch.nn.functional as F


# Depthwise Dilated Separable Convolution (DDSConv) module
class DDSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=2):
        super(DDSConv, self).__init__()
        self.dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=out_channels)
        self.pointwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.dilated_conv(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


# Deconvolution layer for upsampling
class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DeconvLayer, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding)

    def forward(self, x):
        return self.deconv(x)


# DenseNet block, a series of layers that concatenate their outputs
class DenseNet(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


# CBAM module (Convolutional Block Attention Module)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# channel attention module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


# spatial attention module
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x * self.sigmoid(x)


# Residual CBAM block
class ResCBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResCBAM, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv(x))
        x = self.cbam(x)
        x = x + residual
        return x


# LFNet model definition
class LFNet(nn.Module):
    def __init__(self):
        super(LFNet, self).__init__()
        self.ddsconv1 = DDSConv(3, 64)
        self.ddsconv2 = DDSConv(64, 128)
        self.ddsconv3 = DDSConv(1152, 2304)
        self.ddsconv4 = DDSConv(32, 32)
        self.ddsconv5 = DDSConv(576, 32)
        self.ddsconv6 = DDSConv(128, 32)
        self.ddsconv7 = DDSConv(64, 32)
        self.dense_block1 = DenseNet(128, 64, 4)
        self.dense_block2 = DenseNet(384, 64, 4)
        self.dense_block3 = DenseNet(640, 64, 4)
        self.dense_block4 = DenseNet(896, 64, 4)
        self.deconv1 = DeconvLayer(2304, 576)
        self.deconv2 = DeconvLayer(576, 128)
        self.deconv3 = DeconvLayer(128, 64)
        self.deconv4 = DeconvLayer(64, 32)
        self.rescbam1 = ResCBAM(64, 64)
        self.rescbam2 = ResCBAM(128, 128)
        self.rescbam3 = ResCBAM(2304, 2304)
        self.rescbam4 = ResCBAM(576, 576)
        self.rescbam5 = ResCBAM(128, 128)
        self.rescbam6 = ResCBAM(64, 64)
        self.rescbam7 = ResCBAM(32, 32)
        self.res_block = ResidualBlock(2304, 32)
        self.contour_attention = ContourAttention(32, 32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.ddsconv1(x)
        print("After ddsconv1:", x1.shape)

        x2 = self.rescbam1(x1)
        print("After rescbam1:", x2.shape)

        x3 = self.ddsconv2(x2)
        print("After ddsconv2:", x3.shape)

        x4 = self.rescbam2(x3)
        print("After rescbam2:", x4.shape)

        dense_out = self.dense_block1(x4)
        print("After dense_block1:", dense_out.shape)
        dense_out = self.dense_block2(dense_out)
        print("After dense_block2:", dense_out.shape)
        dense_out = self.dense_block3(dense_out)
        print("After dense_block3:", dense_out.shape)
        dense_out = self.dense_block4(dense_out)
        print("After dense_block4:", dense_out.shape)

        x5 = self.ddsconv3(dense_out)
        print("After ddsconv3:", x5.shape)

        x6 = self.rescbam3(x5)
        print("After rescbam3:", x6.shape)

        x7 = self.deconv1(x6)
        print("After deconv1:", x7.shape)

        x8 = self.rescbam4(x7)
        print("After rescbam4:", x8.shape)

        x9 = self.deconv2(x8)
        print("After deconv2:", x9.shape)

        x10 = self.rescbam5(x9)
        print("After rescbam5:", x10.shape)

        x11 = self.deconv3(x10)
        print("After deconv3:", x11.shape)

        x12 = self.rescbam6(x11)
        print("After rescbam6:", x12.shape)

        x13 = self.deconv4(x12)
        print("After deconv4:", x13.shape)

        x14 = self.rescbam7(x13)
        print("After rescbam7:", x14.shape)

        x15 = self.ddsconv4(x14)
        print("After ddsconv4:", x15.shape)

        x16 = self.res_block(x6)
        print("After res_block:", x16.shape)

        x17 = self.ddsconv5(x8)
        print("After ddsconv5:", x17.shape)

        x18 = self.ddsconv6(x10)
        print("After ddsconv6:", x18.shape)

        x19 = self.ddsconv7(x12)
        print("After ddsconv7:", x19.shape)

        x_contour_attention = self.contour_attention(x0=x16, x1=x17, x2=x18, x3=x19, x4=x15)
        print("After contour_attention:", x_contour_attention.shape)

        # Final output
        x14 = F.interpolate(x14, size=(128, 128))
        out = self.final_conv(x14 + x_contour_attention)
        return out
