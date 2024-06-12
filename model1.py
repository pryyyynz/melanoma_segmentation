import torch
import torch.nn as nn
import torch.nn.functional as F


class DDSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=2):
        super(DDSConv, self).__init__()
        self.dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                        groups=out_channels)
        self.pointwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.dilated_conv(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DeconvLayer, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding)

    def forward(self, x):
        return self.deconv(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
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


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


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


class ContourAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContourAttention, self).__init__()
        self.gated_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gated_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.res_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        gated = F.relu(self.gated_conv1(x))
        gated = self.gated_conv2(gated)
        res = F.relu(self.res_conv1(x))
        res = self.res_conv2(res)
        return self.conv1x1(gated + res)


class LFNet(nn.Module):
    def __init__(self):
        super(LFNet, self).__init__()
        self.encoder1 = DDSConv(3, 64)
        self.encoder2 = DDSConv(64, 128)
        self.encoder3 = DDSConv(32, 32)
        self.encoder4 = DDSConv(1152, 2304)
        self.dense_block1 = DenseBlock(128, 64, 4)
        self.dense_block2 = DenseBlock(384, 64, 4)
        self.dense_block3 = DenseBlock(640, 64, 4)
        self.dense_block4 = DenseBlock(896, 64, 4)
        self.res_block = ResNetBlock(32, 32)
        self.deconv1 = DeconvLayer(2304, 572)
        self.deconv2 = DeconvLayer(64, 32)
        self.deconv3 = DeconvLayer(572, 128)
        self.deconv4 = DeconvLayer(128, 64)
        self.rescbam1 = ResCBAM(64, 64)
        self.rescbam2 = ResCBAM(128, 128)
        self.rescbam3 = ResCBAM(2304, 2304)
        self.rescbam4 = ResCBAM(572, 572)
        self.rescbam5 = ResCBAM(128, 128)
        self.rescbam6 = ResCBAM(64, 64)
        self.rescbam7 = ResCBAM(32, 32)
        self.contour_attention = ContourAttention(32, 32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        print("Input shape:", x.shape)

        # ddsconv
        x1 = self.encoder1(x)
        print("After encoder1:", x1.shape)

        # rescbam
        x5 = self.rescbam1(x1)
        print("After rescbam1:", x5.shape)

        # ddsconv
        x2 = self.encoder2(x5)
        print("After encoder2:", x2.shape)

        # rescbam
        x6 = self.rescbam2(x2)
        print("After rescbam2:", x6.shape)

        # densenet x4
        dense_out = self.dense_block1(x6)
        print("After dense_block:", dense_out.shape)
        dense_out = self.dense_block2(dense_out)
        print("After dense_block:", dense_out.shape)
        dense_out = self.dense_block3(dense_out)
        print("After dense_block:", dense_out.shape)
        dense_out = self.dense_block4(dense_out)
        print("After dense_block:", dense_out.shape)

        # ddsconv
        x3 = self.encoder4(dense_out)
        print("After encoder4:", x3.shape)

        # rescbam
        x7 = self.rescbam3(x3)
        print("After rescbam3:", x7.shape)

        # deconv
        x5 = self.deconv1(x7)
        print("After deconv1:", x5.shape)

        # rescbam
        x8 = self.rescbam4(x5)
        print("After rescbam4:", x8.shape)

        # deconv
        x7 = self.deconv3(x8)
        print("After deconv3:", x7.shape)

        # rescbam
        x8 = self.rescbam5(x7)
        print("After rescbam5:", x8.shape)

        # deconv
        x8 = self.deconv4(x8)
        print("After deconv4:", x8.shape)

        # rescbam
        x8 = self.rescbam6(x8)
        print("After rescbam6:", x8.shape)

        # deconv
        x8 = self.deconv2(x8)
        print("After deconv2:", x8.shape)

        # rescbam
        x8 = self.rescbam7(x8)
        print("After rescbam7:", x8.shape)

        # ddsconv
        x3 = self.encoder3(x8)
        print("After encoder3:", x3.shape)

        res_out = self.res_block(x3)
        print("After res_block:", res_out.shape)
        x9 = self.contour_attention(res_out)
        print("After contour_attention:", x9.shape)
        out = self.final_conv(x9)
        print("Output shape:", out.shape)
        return out



