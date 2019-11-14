"""The network definition that was used for a second place solution at the DeepGlobe Building Detection challenge."""

import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torch.nn import functional as F


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class UNetDenseNet(nn.Module):
    def __init__(
        self,
        encoder_depth,
        num_classes=2,
        num_filters=32,
        dropout_2d=0.2,
        pretrained=True,
        is_deconv=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 121:
            self.encoder = torchvision.models.densenet121(
                pretrained=pretrained
            ).features
            bottom_channel_nr = 1024
        elif encoder_depth == 161:
            self.encoder = torchvision.models.densenet161(
                pretrained=pretrained
            ).features
            bottom_channel_nr = 2208
        elif encoder_depth == 169:
            self.encoder = torchvision.models.densenet169(
                pretrained=pretrained
            ).features
            bottom_channel_nr = 1664
        elif encoder_depth == 201:
            self.encoder = torchvision.models.densenet201(
                pretrained=pretrained
            ).features
            bottom_channel_nr = 1920
        else:
            raise NotImplementedError(
                "only 121, 161, 169, 201 version of Densenet are implemented"
            )

        self.conv1 = nn.Sequential(
            self.encoder.conv0,
            self.encoder.norm0,
            self.encoder.relu0,
            self.encoder.pool0,
        )

        self.conv2 = self.encoder.denseblock1
        self.conv3 = self.encoder.denseblock2
        self.conv4 = self.encoder.denseblock3
        self.conv5 = self.encoder.denseblock4

        self.tr2 = self.encoder.transition1
        self.tr3 = self.encoder.transition2
        self.tr4 = self.encoder.transition3
        self.norm5 = self.encoder.norm5

        self.center = DecoderCenter(
            bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, False
        )

        self.dec5 = DecoderBlockV(
            bottom_channel_nr + num_filters * 8,
            num_filters * 8 * 2,
            num_filters * 2,
            is_deconv,
        )
        self.dec4 = DecoderBlockV(1856, num_filters * 8, num_filters * 2, is_deconv)
        self.dec3 = DecoderBlockV(576, num_filters * 4, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV(320, num_filters * 2, num_filters * 2, is_deconv)
        self.dec1 = DecoderBlockV(
            num_filters * 2, num_filters, num_filters * 2, is_deconv
        )
        self.dec0 = ConvRelu(num_filters * 10, num_filters * 2)
        self.final = nn.Conv2d(num_filters * 2, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        t2 = self.tr2(conv2)
        conv3 = self.conv3(t2)
        t3 = self.tr3(conv3)

        conv4 = self.conv4(t3)
        t4 = self.tr4(conv4)

        conv5 = self.conv5(t4)
        conv5 = self.norm5(conv5)

        center = self.center(conv5)
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)

        f = torch.cat(
            (
                dec1,
                F.upsample(dec2, scale_factor=2, mode="bilinear", align_corners=False),
                F.upsample(dec3, scale_factor=4, mode="bilinear", align_corners=False),
                F.upsample(dec4, scale_factor=8, mode="bilinear", align_corners=False),
                F.upsample(dec5, scale_factor=16, mode="bilinear", align_corners=False),
            ),
            1,
        )

        dec0 = self.dec0(F.dropout2d(f, p=self.dropout_2d))

        return self.final(dec0)


class DecoderBlockV(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels, out_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class DecoderCenter(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderCenter, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels, out_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)
