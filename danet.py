from __future__ import division
import torch
import torch.nn as nn
from torch.nn.functional import upsample, interpolate
from da_att import PAM_Module, CAM_Module

__all__ = ['DANet', 'get_danet']

class DANet(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d):
        super(DANet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=3, stride=1, padding=1),
            norm_layer(128),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=3, stride=1, padding=1),
            norm_layer(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1),
            norm_layer(128),
            nn.ReLU()
        )
        self.head = DANetHead(384, nclass, norm_layer)

    def forward(self, orig_label, orig_pred, label_pred):
        x1 = torch.cat(orig_label, dim=1)  # (RGB + label) -> (batch, 4, H, W)
        x2 = torch.cat(orig_pred, dim=1)   # (RGB + pred) -> (batch, 4, H, W)
        x3 = torch.cat(label_pred, dim=1)  # (label + pred) -> (batch, 2, H, W)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.head(x)
        x = list(x)
        imsize = orig_label[0].size()[2:]
        x[0] = interpolate(x[0], imsize, mode='bilinear', align_corners=True)

        return x[0]

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        exter_channels = in_channels // 6
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(exter_channels)
        self.sc = CAM_Module(exter_channels)

        self.conv6a = nn.Sequential(nn.Conv2d(inter_channels, exter_channels, 3, padding=1, bias=False),
                                    norm_layer(exter_channels),
                                    nn.ReLU())
        self.conv6c = nn.Sequential(nn.Conv2d(inter_channels, exter_channels, 3, padding=1, bias=False),
                                    norm_layer(exter_channels),
                                    nn.ReLU())

        self.conv7a = nn.Sequential(nn.Conv2d(exter_channels, exter_channels, 3, padding=1, bias=False),
                                    norm_layer(exter_channels),
                                    nn.ReLU())
        self.conv7c = nn.Sequential(nn.Conv2d(exter_channels, exter_channels, 3, padding=1, bias=False),
                                    norm_layer(exter_channels),
                                    nn.ReLU())

        self.conv8a = nn.Sequential(nn.Conv2d(exter_channels, exter_channels, 3, padding=1, bias=False),
                                    norm_layer(exter_channels),
                                    nn.ReLU())
        self.conv8c = nn.Sequential(nn.Conv2d(exter_channels, exter_channels, 3, padding=1, bias=False),
                                    norm_layer(exter_channels),
                                    nn.ReLU())
        self.conv9a = nn.Sequential(
            nn.Dropout2d(0.1, False), nn.Conv2d(exter_channels, out_channels, 1)
        )
        self.conv9c = nn.Sequential(
            nn.Dropout2d(0.1, False), nn.Conv2d(exter_channels, out_channels, 1)
        )

        self.conv11 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(exter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.conv6a(feat1)
        sa_conv1 = self.conv7a(sa_feat)
        sa_conv2 = self.conv8a(sa_conv1)
        sa_conv3 = self.sa(sa_conv2)
        sa_output = self.conv9a(sa_conv3)

        feat2 = self.conv5c(x)
        sc_feat2 = self.conv6c(feat2)
        sc_conv1 = self.conv7c(sc_feat2)
        sc_conv2 = self.conv8c(sc_conv1)
        sc_conv3 = self.sc(sc_conv2)
        sc_output = self.conv9c(sc_conv3)

        feat_sum = sa_conv3 + sc_conv3

        sasc_output = self.conv11(feat_sum)

        return [sasc_output, sa_output, sc_output]


def get_danet(nclass, pretrained=False, **kwargs):
    model = DANet(nclass=nclass, **kwargs)
    if pretrained:
        # Load the pre-trained model weights
        pass
    return model
