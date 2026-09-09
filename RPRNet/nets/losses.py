"""Training losses for RPR-Net."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModifiedFocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
        target = target.float().to(prediction.device)
        positive = target.eq(1).float()
        negative = target.lt(1).float()
        negative_weights = torch.pow(1 - target, self.beta)
        positive_loss = (
            torch.log(prediction) * torch.pow(1 - prediction, self.alpha) * positive
        ).sum()
        negative_loss = (
            torch.log(1 - prediction)
            * torch.pow(prediction, self.alpha)
            * negative_weights
            * negative
        ).sum()
        positive_count = positive.sum()
        if positive_count.item() == 0:
            return -negative_loss
        return -(positive_loss + negative_loss) / positive_count


class RegL1Loss(nn.Module):
    def forward(self, prediction, target, mask):
        mask = mask.expand_as(prediction).float()
        loss = F.l1_loss(prediction * mask, target * mask, reduction="sum")
        return loss / torch.clamp(mask.sum(), min=1e-4)


class RPRNetLoss(nn.Module):
    def __init__(
        self,
        heatmap_weight=1.0,
        radius_weight=0.1,
        offset_weight=1.0,
        focal_alpha=2.0,
        focal_beta=4.0,
    ):
        super().__init__()
        self.heatmap_loss = ModifiedFocalLoss(alpha=focal_alpha, beta=focal_beta)
        self.regression_loss = RegL1Loss()
        self.heatmap_weight = heatmap_weight
        self.radius_weight = radius_weight
        self.offset_weight = offset_weight

    def forward(self, prediction, target):
        heatmap_loss = self.heatmap_loss(prediction["heatmap"].sigmoid(), target["heatmap"])
        radius_loss = self.regression_loss(
            prediction["radius_map"], target["radius_map"], target["reg_mask"]
        )
        offset_loss = self.regression_loss(
            prediction["offset_map"], target["offset_map"], target["reg_mask"]
        )
        total_loss = (
            self.heatmap_weight * heatmap_loss
            + self.radius_weight * radius_loss
            + self.offset_weight * offset_loss
        )
        return total_loss, {
            "total_loss": total_loss,
            "heatmap_loss": heatmap_loss,
            "radius_loss": radius_loss,
            "offset_loss": offset_loss,
        }
