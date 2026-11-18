"""DeepLabV3+ with a ResNet-50 encoder (Chen et al., ECCV 2018; workbook P002).

Architecture (output stride 16):

* Encoder: torchvision ResNet-50, ImageNet-pretrained. ``layer4`` uses
  dilation 2 instead of stride 2, so the deepest feature map is 1/16 of the
  input (32x32 for a 512x512 tile, 2048 channels). ``layer1`` (1/4, 256
  channels) supplies low-level features.
* ASPP: a 1x1 branch, three 3x3 atrous branches (rates 6, 12, 18) and an
  image-pooling branch, each producing 256 channels, concatenated and fused
  by a 1x1 conv followed by dropout (0.1 by default).
* Decoder: low-level features reduced to 48 channels by 1x1 conv, ASPP output
  upsampled 4x and concatenated, two 3x3 convs (256 channels), a 1x1
  classifier, and a final 4x bilinear upsampling to input resolution.

``separable=True`` replaces the 3x3 convolutions in ASPP and decoder with
depthwise-separable convolutions, as in the paper's title; ``False`` uses
standard convolutions (a common ResNet variant).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


def conv_bn_relu(cin: int, cout: int, k: int = 3, dilation: int = 1, separable: bool = False) -> nn.Sequential:
    """Build a conv -> BN -> ReLU block.

    Args:
        cin: Input channels.
        cout: Output channels.
        k: Kernel size (1 or 3).
        dilation: Atrous rate for the spatial convolution.
        separable: If True and ``k > 1``, use a depthwise k x k conv followed
            by a pointwise 1x1 conv (each followed by BN; ReLU at the end).

    Returns:
        ``nn.Sequential`` block preserving spatial size.
    """
    pad = dilation * (k - 1) // 2
    if separable and k > 1:
        return nn.Sequential(
            nn.Conv2d(cin, cin, k, padding=pad, dilation=dilation, groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with an image-level pooling branch."""

    def __init__(self, cin: int, cout: int = 256, rates=(6, 12, 18), separable: bool = True, dropout: float = 0.1):
        """Create the parallel branches and the 1x1 projection.

        Args:
            cin: Encoder output channels (2048 for ResNet-50).
            cout: Channels per branch and of the output.
            rates: Atrous rates of the three 3x3 branches (for output stride 16).
            separable: Use depthwise-separable 3x3 convolutions.
            dropout: Dropout probability after the projection.
        """
        super().__init__()
        self.branches = nn.ModuleList(
            [conv_bn_relu(cin, cout, 1)] + [conv_bn_relu(cin, cout, 3, r, separable) for r in rates]
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1, bias=False),
                                  nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.project = nn.Sequential(conv_bn_relu(cout * (len(rates) + 2), cout, 1), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate all branch outputs (pooling branch upsampled) and project."""
        feats = [b(x) for b in self.branches]
        feats.append(F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.project(torch.cat(feats, dim=1))


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ semantic segmentation network with a ResNet-50 encoder."""

    def __init__(self, num_classes: int, pretrained: bool = True, separable: bool = True,
                 aspp_rates=(6, 12, 18), low_level_channels: int = 48, aspp_dropout: float = 0.1):
        """Build encoder, ASPP and decoder.

        Args:
            num_classes: Number of output classes.
            pretrained: Load torchvision ``IMAGENET1K_V1`` ResNet-50 weights.
            separable: Depthwise-separable 3x3 convs in ASPP and decoder.
            aspp_rates: ASPP atrous rates.
            low_level_channels: Channels of the reduced ``layer1`` features.
            aspp_dropout: Dropout after the ASPP projection.
        """
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        r = resnet50(weights=weights, replace_stride_with_dilation=[False, False, True])
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.aspp = ASPP(2048, 256, aspp_rates, separable, aspp_dropout)
        self.reduce = conv_bn_relu(256, low_level_channels, 1)
        self.decoder = nn.Sequential(
            conv_bn_relu(256 + low_level_channels, 256, 3, separable=separable),
            conv_bn_relu(256, 256, 3, separable=separable),
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def encoder_parameters(self):
        """Parameters of the pretrained ResNet-50 (typically trained with a lower LR)."""
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from m.parameters()

    def head_parameters(self):
        """Parameters of the randomly initialised ASPP, decoder and classifier."""
        for m in (self.aspp, self.reduce, self.decoder, self.classifier):
            yield from m.parameters()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 256-channel decoder feature map at 1/4 input resolution."""
        low = self.layer1(self.stem(x))
        high = self.aspp(self.layer4(self.layer3(self.layer2(low))))
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.decoder(torch.cat([high, self.reduce(low)], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-pixel class logits at input resolution, shape ``(N, K, H, W)``."""
        logits = self.classifier(self.forward_features(x))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(cfg: dict) -> DeepLabV3Plus:
    """Construct the model from the ``model`` and ``dataset`` config sections."""
    m = cfg.get("model", {})
    return DeepLabV3Plus(
        num_classes=cfg["dataset"]["num_classes"],
        pretrained=m.get("pretrained", True),
        separable=m.get("separable", True),
        aspp_rates=tuple(m.get("aspp_rates", (6, 12, 18))),
        aspp_dropout=m.get("aspp_dropout", 0.1),
    )
