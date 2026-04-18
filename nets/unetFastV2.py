"""
MSFA_Unet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from .ASPP import ASPP


class MultiScaleAggregation(nn.Module):
    def __init__(self, feat_channels, dim_out=64, bn_mom=0.1):
        super(MultiScaleAggregation, self).__init__()
        c1, c3, c5 = feat_channels
        self.proj1_gap = nn.AdaptiveAvgPool2d(1)
        self.proj1_conv = nn.Sequential(
            nn.Conv2d(c1, dim_out, 1, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(c3, dim_out, 1, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.proj5 = nn.Sequential(
            nn.Conv2d(c5, dim_out, 1, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dim_out * 3, dim_out, 1, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat1, feat3, feat5):
        H, W = feat3.size(2), feat3.size(3)
        gap = self.proj1_gap(feat1)
        f1 = self.proj1_conv(gap)
        f1 = F.interpolate(f1, size=(H, W), mode="bilinear", align_corners=True)
        f3 = self.proj3(feat3)
        f5 = F.interpolate(self.proj5(feat5), size=(H, W), mode="bilinear", align_corners=True)
        out = torch.cat([f1, f3, f5], dim=1)
        out = self.fuse(out)
        return out


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        up_feat = self.up(inputs2)
        outputs = torch.cat([inputs1, up_feat], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone="mobilenetv3_small"):
        super(Unet, self).__init__()
        if backbone == "mobilenetv3_large":
            self.mobilenet = mobilenetv3_large(pretrained=pretrained)
            in_filters = [144, 272, 552, 1072]
            feat_channels = [16, 40, 960]
        elif backbone == "mobilenetv3_small":
            self.mobilenet = mobilenetv3_small(pretrained=pretrained)
            in_filters = [144, 272, 536, 624]
            feat_channels = [16, 24, 576]
        else:
            raise ValueError(
                "This package only supports `backbone=mobilenetv3_large` or `mobilenetv3_small`,"
                "Current: `{}`".format(backbone)
            )
        out_filters = [64, 128, 256, 512]
        msa_dim_out = out_filters[0]
        self.msa = MultiScaleAggregation(feat_channels, dim_out=msa_dim_out)
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0] + msa_dim_out, out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.aspp_up4 = ASPP(dim_in=out_filters[3], dim_out=out_filters[3])
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.backbone = backbone

    def forward(self, inputs):
        feat1, feat2, feat3, feat4, feat5 = self.mobilenet.forward(inputs)
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        msa_feat = self.msa(feat1, feat3, feat5)
        msa_feat = F.interpolate(
            msa_feat, size=(up1.size(2), up1.size(3)), mode="bilinear", align_corners=True
        )
        up1 = self.up_conv(torch.cat([up1, msa_feat], dim=1))
        return self.final(up1)

    def freeze_backbone(self):
        for param in self.mobilenet.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.mobilenet.parameters():
            param.requires_grad = True
