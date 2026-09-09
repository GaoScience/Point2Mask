"""Checkpoint I/O for RPR-Net training and inference."""

from pathlib import Path

import torch


def extract_state_dict(payload):
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(payload)}")
    for key in ("ema_state_dict", "model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def load_model_weights(model, path, device="cpu", strict=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = extract_state_dict(payload)
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    incompatible = model.load_state_dict(cleaned, strict=strict)
    return incompatible


def save_training_checkpoint(model, optimizer, scheduler, epoch, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )


def save_model_weights(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def resume_training(model, optimizer, scheduler, path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(extract_state_dict(payload), strict=True)
    if "optimizer_state_dict" not in payload:
        raise ValueError("Resume checkpoint does not contain optimizer state")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    return int(payload.get("epoch", 0))
