"""Editable RPR-Net training settings for SIRSTD."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RPRNetTrainConfig:
    # Model and online target generation
    input_size: int = 512
    num_frames: int = 3
    down_scale: int = 4
    width_multiplier: float = 0.5
    depth_multiplier: float = 0.33

    # Optimization
    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    seed: int = 3407
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    momentum: float = 0.9
    use_amp: bool = False

    # Loss and checkpointing
    heatmap_loss_weight: float = 1.0
    radius_loss_weight: float = 0.1
    offset_loss_weight: float = 1.0
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    save_every: int = 10

    def validate(self):
        if self.input_size <= 0 or self.input_size % 16 != 0:
            raise ValueError("input_size must be a positive multiple of 16")
        if self.num_frames not in (1, 3):
            raise ValueError("num_frames must be 1 or 3")
        if self.down_scale != 4:
            raise ValueError("down_scale must be 4 for the current RPR-Net architecture")
        if self.width_multiplier <= 0 or self.depth_multiplier <= 0:
            raise ValueError("model width and depth multipliers must be positive")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be at least 1")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.optimizer.lower() not in {"adamw", "adam", "sgd"}:
            raise ValueError("optimizer must be adamw, adam, or sgd")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("learning rates are invalid")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.momentum < 0:
            raise ValueError("momentum cannot be negative")
        if min(
            self.heatmap_loss_weight,
            self.radius_loss_weight,
            self.offset_loss_weight,
        ) < 0:
            raise ValueError("loss weights cannot be negative")
        if self.focal_alpha < 0 or self.focal_beta < 0:
            raise ValueError("focal_alpha and focal_beta cannot be negative")
        if self.save_every < 1:
            raise ValueError("save_every must be at least 1")


TRAIN_CONFIG = RPRNetTrainConfig()
