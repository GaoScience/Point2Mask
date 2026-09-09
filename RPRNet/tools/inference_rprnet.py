"""Run RPR-Net on one temporal sequence or a split-file-defined dataset."""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RPRNet.datasets.rprnet_dataset import (
    RPRNetPredictionDataset,
    find_image_path,
    preprocess,
)
from RPRNet.nets.rpr_net import RPRNet
from RPRNet.utils.checkpoint import load_model_weights
from RPRNet.utils.decoder import (
    decode_centers,
    detection_records,
    render_pamg_mask,
    render_radius_mask,
)


# -----------------------------------------------------------------------------
# Editable inference settings
# -----------------------------------------------------------------------------
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "rprnet_sirstd_v1.pth"
DEFAULT_OUTPUT_DIR = "outputs/inference"
DEFAULT_DEVICE = ""
DEFAULT_MASK_MODE = "pamg"

NUM_FRAMES = 3
INPUT_SIZE = 512
MODEL_OUTPUT_STRIDE = 4
WIDTH_MULTIPLIER = 0.5
DEPTH_MULTIPLIER = 0.33

BATCH_SIZE = 4
NUM_WORKERS = 4
SCORE_THRESHOLD = 0.1
TOPK = 100
SAVE_RAW_OUTPUTS = False

PAMG_RS_SCALE = 5.0
PAMG_RS_CAP = 20.0
PAMG_POLARITY = 0


def parse_args():
    parser = argparse.ArgumentParser(description="RPR-Net inference")
    parser.add_argument("--checkpoint", default=str(DEFAULT_WEIGHTS))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", nargs="+", help="Ordered frames for one sequence")
    source.add_argument("--sample-list", help="Text file containing dataset sample IDs")
    parser.add_argument("--image-root", help="Image directory used with --sample-list")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mode",
        choices=("radius", "pamg"),
        default=None,
        help=f"Override DEFAULT_MASK_MODE (currently: {DEFAULT_MASK_MODE})",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="For example: cuda:0 or cpu")
    args = parser.parse_args()

    if args.sample_list and not args.image_root:
        parser.error("--image-root is required with --sample-list")
    return args


def validate_inference_settings():
    if NUM_FRAMES not in (1, 3):
        raise ValueError("NUM_FRAMES must be 1 or 3")
    if INPUT_SIZE <= 0 or INPUT_SIZE % 16 != 0:
        raise ValueError("INPUT_SIZE must be a positive multiple of 16")
    if MODEL_OUTPUT_STRIDE != 4:
        raise ValueError("MODEL_OUTPUT_STRIDE must be 4 for the current RPR-Net")
    if WIDTH_MULTIPLIER <= 0 or DEPTH_MULTIPLIER <= 0:
        raise ValueError("model width and depth multipliers must be positive")
    if BATCH_SIZE < 1 or NUM_WORKERS < 0:
        raise ValueError("BATCH_SIZE must be positive and NUM_WORKERS cannot be negative")
    if TOPK < 1 or not 0.0 <= SCORE_THRESHOLD <= 1.0:
        raise ValueError("TOPK or SCORE_THRESHOLD is invalid")
    if DEFAULT_MASK_MODE not in {"radius", "pamg"}:
        raise ValueError("DEFAULT_MASK_MODE must be radius or pamg")
    if PAMG_RS_SCALE <= 0:
        raise ValueError("PAMG_RS_SCALE must be positive")
    if PAMG_RS_CAP is not None and PAMG_RS_CAP <= 0:
        raise ValueError("PAMG_RS_CAP must be positive or None")
    if PAMG_POLARITY not in {-1, 0, 1}:
        raise ValueError("PAMG_POLARITY must be -1, 0, or 1")


def select_device(argument):
    if argument:
        return torch.device(argument)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_sequence(image_paths, input_size):
    frames = []
    center_gray = None
    original_hw = None
    middle = len(image_paths) // 2
    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        if index == middle:
            center_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            original_hw = center_gray.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            image_rgb,
            (input_size, input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        frames.append(torch.from_numpy(preprocess(resized)))
    return torch.stack(frames, dim=0).unsqueeze(0).float(), center_gray, original_hw


def write_records(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("row", "col", "radius", "score"))
        writer.writeheader()
        for record in records:
            writer.writerow({key: f"{record[key]:.6f}" for key in writer.fieldnames})


def run_model(model, images, device):
    outputs = model(images.to(device, non_blocking=True))
    heatmap = outputs["heatmap"].sigmoid()
    points = decode_centers(
        heatmap,
        outputs["offset_map"],
        topk=TOPK,
        score_threshold=SCORE_THRESHOLD,
    )
    return outputs, heatmap, points


def render_mask(records, center_gray, mode):
    if mode == "radius":
        return render_radius_mask(records, center_gray.shape)
    return render_pamg_mask(
        records,
        center_gray,
        Rs_scale=PAMG_RS_SCALE,
        Rs_cap=PAMG_RS_CAP,
        polarity=PAMG_POLARITY,
    )


def save_raw_maps(raw_dir, stem, outputs, heatmap, batch_index):
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.save(raw_dir / f"{stem}_heatmap.npy", heatmap[batch_index].detach().cpu().numpy())
    np.save(
        raw_dir / f"{stem}_offset_map.npy",
        outputs["offset_map"][batch_index].detach().cpu().numpy(),
    )
    np.save(
        raw_dir / f"{stem}_radius_map.npy",
        outputs["radius_map"][batch_index].detach().cpu().numpy(),
    )


def infer_one_sequence(model, args, device, mode):
    image_paths = [Path(path) for path in args.images]
    if len(image_paths) == 1 and NUM_FRAMES == 3:
        image_paths *= 3
    if len(image_paths) != NUM_FRAMES:
        raise ValueError(f"Expected {NUM_FRAMES} frames, received {len(image_paths)}")

    images, center_gray, original_hw = load_sequence(image_paths, INPUT_SIZE)
    with torch.inference_mode():
        outputs, heatmap, points_batch = run_model(model, images, device)
    records = detection_records(
        points_batch[0],
        outputs["radius_map"],
        batch_index=0,
        original_hw=original_hw,
        input_size=INPUT_SIZE,
        down_scale=MODEL_OUTPUT_STRIDE,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_records(records, output_dir / "detections.csv")
    mask = render_mask(records, center_gray, mode)
    if not cv2.imwrite(str(output_dir / f"mask_{mode}.png"), mask):
        raise OSError(f"Failed to save mask in {output_dir}")
    if SAVE_RAW_OUTPUTS:
        save_raw_maps(output_dir / "raw", "prediction", outputs, heatmap, 0)
    print(f"Saved {len(records)} detections to {output_dir}")


def infer_dataset(model, args, device, mode):
    dataset = RPRNetPredictionDataset(
        split_file=args.sample_list,
        image_root=args.image_root,
        input_size=INPUT_SIZE,
        num_frames=NUM_FRAMES,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    output_dir = Path(args.output_dir)
    detection_dir = output_dir / "detections"
    mask_dir = output_dir / f"masks_{mode}"
    raw_dir = output_dir / "raw"
    detection_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for images, sample_ids in tqdm(loader, desc="Inferring", ncols=100):
            outputs, heatmap, points_batch = run_model(model, images, device)
            for batch_index, sample_id in enumerate(sample_ids):
                center_path = find_image_path(args.image_root, sample_id)
                if center_path is None:
                    raise FileNotFoundError(f"Image not found for sample: {sample_id}")
                center_gray = cv2.imread(str(center_path), cv2.IMREAD_GRAYSCALE)
                if center_gray is None:
                    raise OSError(f"Failed to read image: {center_path}")
                records = detection_records(
                    points_batch[batch_index],
                    outputs["radius_map"],
                    batch_index=batch_index,
                    original_hw=center_gray.shape,
                    input_size=INPUT_SIZE,
                    down_scale=MODEL_OUTPUT_STRIDE,
                )
                stem = Path(sample_id).stem
                write_records(records, detection_dir / f"{stem}.csv")
                mask = render_mask(records, center_gray, mode)
                if not cv2.imwrite(str(mask_dir / f"{stem}.png"), mask):
                    raise OSError(f"Failed to save mask for sample: {sample_id}")
                if SAVE_RAW_OUTPUTS:
                    save_raw_maps(raw_dir, stem, outputs, heatmap, batch_index)

    print(f"Prediction files saved to {output_dir}")


def main():
    args = parse_args()
    validate_inference_settings()
    mode = args.mode or DEFAULT_MASK_MODE
    device = select_device(args.device)
    model = RPRNet(
        num_frames=NUM_FRAMES,
        wid_mul=WIDTH_MULTIPLIER,
        dep_mul=DEPTH_MULTIPLIER,
    ).to(device)
    load_model_weights(model, args.checkpoint, device=device, strict=True)
    model.eval()

    if args.images:
        infer_one_sequence(model, args, device, mode)
    else:
        infer_dataset(model, args, device, mode)


if __name__ == "__main__":
    main()
