"""Minimal RPR-Net training loop."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RPRNet.datasets.rprnet_dataset import RPRNetTrainDataset
from RPRNet.configs.rprnet_sirstd import TRAIN_CONFIG
from RPRNet.nets.losses import RPRNetLoss
from RPRNet.nets.rpr_net import RPRNet
from RPRNet.utils.checkpoint import (
    load_model_weights,
    resume_training,
    save_model_weights,
    save_training_checkpoint,
)
from RPRNet.utils.misc import build_optimizer, setup_seed, worker_init_fn

try:
    from torch.amp import GradScaler, autocast

    def autocast_context(enabled):
        return autocast(device_type="cuda", enabled=enabled)

except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    def autocast_context(enabled):
        return autocast(enabled=enabled)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RPR-Net on a training split")
    parser.add_argument("--train-list", required=True, help="Text file containing training sample IDs")
    parser.add_argument("--image-root", required=True, help="Directory containing input frames")
    parser.add_argument("--mask-root", required=True, help="Directory containing binary training masks")
    parser.add_argument("--save-dir", default="outputs/rprnet_train")
    parser.add_argument("--pretrained", default="", help="Optional model weights")
    parser.add_argument("--resume", default="", help="Optional training checkpoint")
    parser.add_argument("--device", default="", help="For example: cuda:0 or cpu")
    return parser.parse_args()


def select_device(device_argument):
    if device_argument:
        return torch.device(device_argument)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, criterion, optimizer, scaler, loader, device, amp_enabled, epoch):
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch}", ncols=100)
    for images, targets, _ in progress:
        images = images.to(device, non_blocking=True)
        targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(amp_enabled):
            predictions = model(images)
            loss, loss_items = criterion(predictions, targets)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item())
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            heatmap=f"{loss_items['heatmap_loss'].item():.4f}",
        )
    return running_loss / max(len(loader), 1)


def main():
    args = parse_args()
    config = TRAIN_CONFIG
    config.validate()

    setup_seed(config.seed)
    device = select_device(args.device)
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Training settings: RPRNet/configs/rprnet_sirstd.py")

    dataset = RPRNetTrainDataset(
        split_file=args.train_list,
        image_root=args.image_root,
        mask_root=args.mask_root,
        input_size=config.input_size,
        num_frames=config.num_frames,
        down_scale=config.down_scale,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
        persistent_workers=config.num_workers > 0,
    )

    model = RPRNet(
        num_frames=config.num_frames,
        wid_mul=config.width_multiplier,
        dep_mul=config.depth_multiplier,
    ).to(device)
    criterion = RPRNetLoss(
        heatmap_weight=config.heatmap_loss_weight,
        radius_weight=config.radius_loss_weight,
        offset_weight=config.offset_loss_weight,
        focal_alpha=config.focal_alpha,
        focal_beta=config.focal_beta,
    ).to(device)
    optimizer = build_optimizer(
        model,
        name=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_learning_rate,
    )
    scaler = GradScaler(enabled=amp_enabled)

    start_epoch = 0
    if args.resume:
        start_epoch = resume_training(model, optimizer, scheduler, args.resume, device)
        print(f"Resumed after epoch {start_epoch}")
    elif args.pretrained:
        load_model_weights(model, args.pretrained, device=device, strict=True)
        print(f"Loaded pretrained weights: {args.pretrained}")

    for epoch_index in range(start_epoch, config.epochs):
        epoch = epoch_index + 1
        average_loss = train_one_epoch(
            model,
            criterion,
            optimizer,
            scaler,
            loader,
            device,
            amp_enabled,
            epoch,
        )
        scheduler.step()
        print(f"Epoch {epoch}/{config.epochs} - training loss: {average_loss:.6f}")

        save_training_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            output_dir / "checkpoint_last.pth",
        )
        if epoch % config.save_every == 0 or epoch == config.epochs:
            save_model_weights(model, output_dir / f"rprnet_epoch_{epoch:04d}.pth")


if __name__ == "__main__":
    main()
