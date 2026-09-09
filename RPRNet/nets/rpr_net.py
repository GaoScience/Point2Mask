"""RPR-Net: temporal point and radius regression for infrared targets."""

import torch
import torch.nn as nn


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        return SiLU()
    return nn.ReLU(inplace=inplace)


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        padding = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        del shortcut
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, 1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, 1, act=act)
        self.m = nn.Sequential(
            *[
                nn.Sequential(
                    BaseConv(hidden_channels, hidden_channels, 1, 1, act=act),
                    BaseConv(
                        hidden_channels,
                        hidden_channels,
                        3,
                        1,
                        groups=hidden_channels if depthwise else 1,
                        act=act,
                    ),
                )
                for _ in range(n)
            ]
        )

    def forward(self, x):
        branch_1 = self.m(self.conv1(x))
        branch_2 = self.conv2(x)
        return self.conv3(torch.cat((branch_1, branch_2), dim=1))


class CSPDarknetTinyTrimmed(nn.Module):
    """Compact CSP backbone that retains strides 4, 8, and 16."""

    def __init__(self, dep_mul=0.33, wid_mul=0.5, act="silu"):
        super().__init__()
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        self.stem = BaseConv(3, base_channels, 3, 2, act=act)
        self.dark2 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, act=act),
        )
        self.dark3 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, act=act),
        )
        self.dark4 = nn.Sequential(
            BaseConv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, act=act),
        )

    def forward(self, x):
        x = self.stem(x)
        p2 = self.dark2(x)
        p3 = self.dark3(p2)
        p4 = self.dark4(p3)
        return {"p2": p2, "p3": p3, "p4": p4}


class FPN_3Level(nn.Module):
    """Top-down feature pyramid from stride 16 to stride 4."""

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        c2, c3, c4 = in_channels_list
        self.reduce_p4 = BaseConv(c4, c3, 1, 1)
        self.csp_p3 = CSPLayer(c3 * 2, c3, n=1, shortcut=False)
        self.reduce_p3 = BaseConv(c3, c2, 1, 1)
        self.csp_p2 = CSPLayer(c2 * 2, out_channels, n=1, shortcut=False)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, feats):
        p2, p3, p4 = feats["p2"], feats["p3"], feats["p4"]
        p4_up = self.upsample(self.reduce_p4(p4))
        p3_out = self.csp_p3(torch.cat([p3, p4_up], dim=1))
        p3_up = self.upsample(self.reduce_p3(p3_out))
        return self.csp_p2(torch.cat([p2, p3_up], dim=1))


class AdaptiveFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 3, in_channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels * 3, bias=False),
            nn.Sigmoid(),
        )
        self.out_conv = BaseConv(in_channels, in_channels, 3, 1)

    def forward(self, feats_list):
        stacked = torch.cat(feats_list, dim=1)
        batch, total_channels, _, _ = stacked.shape
        channels = total_channels // 3
        weights = self.fc(self.avg_pool(stacked).view(batch, total_channels))
        weights = weights.view(batch, total_channels, 1, 1)
        w_prev, w_curr, w_next = torch.split(weights, channels, dim=1)
        fused = feats_list[0] * w_prev + feats_list[1] * w_curr + feats_list[2] * w_next
        return self.out_conv(fused)


class TemporalDifferenceAttention(nn.Module):
    """Temporal Difference Attention (TDA) for center-frame enhancement."""

    def __init__(self, in_channels):
        super().__init__()
        self.motion_conv = nn.Sequential(
            BaseConv(in_channels, in_channels, 3, 1),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid(),
        )
        self.gate_scale = nn.Parameter(torch.tensor(2.0))
        self.fusion = AdaptiveFusion(in_channels)
        # Retained to keep the released checkpoint architecture-compatible.
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, prev_feat, curr_feat, next_feat):
        temporal_change = torch.abs(curr_feat - prev_feat) + torch.abs(curr_feat - next_feat)
        global_jitter = torch.abs(next_feat - prev_feat).mean(dim=(1, 2, 3), keepdim=True)
        motion_gate = torch.exp(-self.gate_scale * global_jitter)
        motion_attention = self.motion_conv(temporal_change) * motion_gate
        curr_enhanced = curr_feat * (1.0 + motion_attention)
        fused = self.fusion([prev_feat, curr_enhanced, next_feat])
        return fused + curr_feat


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention + x


class RPRNet(nn.Module):
    """Regress a center heatmap, sub-pixel offsets, and target radii."""

    def __init__(self, num_frames=3, wid_mul=0.5, dep_mul=0.33):
        super().__init__()
        if num_frames not in (1, 3):
            raise ValueError("RPR-Net supports one frame or three temporal frames")

        self.num_frames = num_frames
        self.backbone = CSPDarknetTinyTrimmed(dep_mul=dep_mul, wid_mul=wid_mul)
        base_channels = int(64 * wid_mul)
        channels = [base_channels * (2 ** i) for i in range(1, 4)]
        self.feat_c = channels[0]
        self.neck = FPN_3Level(in_channels_list=channels, out_channels=self.feat_c)
        if self.num_frames > 1:
            self.tda = TemporalDifferenceAttention(in_channels=self.feat_c)

        self.sam = SpatialAttention()
        self.head_hm = nn.Sequential(BaseConv(self.feat_c, 64, 3, 1), nn.Conv2d(64, 1, 1))
        self.head_off = nn.Sequential(BaseConv(self.feat_c, 64, 3, 1), nn.Conv2d(64, 2, 1))
        self.head_wh = nn.Sequential(
            BaseConv(self.feat_c, 64, 3, 1),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward_single_frame(self, x):
        return self.neck(self.backbone(x))

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("RPRNet expects input shaped [B, T, C, H, W]")
        _, frames, _, _, _ = x.shape
        if not torch.jit.is_tracing() and frames != self.num_frames:
            raise ValueError(f"Model expects {self.num_frames} frames, received {frames}")

        frame_features = [self.forward_single_frame(x[:, index]) for index in range(frames)]
        if self.num_frames > 1:
            middle = self.num_frames // 2
            fused = self.tda(
                frame_features[middle - 1],
                frame_features[middle],
                frame_features[middle + 1],
            )
        else:
            fused = frame_features[0]

        final_feature = self.sam(fused)
        return {
            "heatmap": self.head_hm(final_feature),
            "offset_map": self.head_off(final_feature),
            "radius_map": self.head_wh(final_feature),
        }
