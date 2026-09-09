"""Dataset helpers shared by RPR-Net training and prediction."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_SUFFIXES = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")
MASK_SUFFIXES = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def read_sample_ids(split_file):
    split_file = Path(split_file)
    with split_file.open("r", encoding="utf-8-sig") as handle:
        raw_ids = [line.strip() for line in handle if line.strip()]
    sample_ids = []
    seen = set()
    for raw_id in raw_ids:
        sample_path = Path(raw_id)
        if sample_path.is_absolute() or sample_path.parent != Path("."):
            raise ValueError(
                f"Sample IDs in {split_file} must be file names, not paths: {raw_id}"
            )
        sample_id = (
            sample_path.stem
            if sample_path.suffix.lower() in IMAGE_SUFFIXES
            else raw_id
        )
        if sample_id in seen:
            raise ValueError(f"Duplicate sample ID in {split_file}: {sample_id}")
        seen.add(sample_id)
        sample_ids.append(sample_id)
    if not sample_ids:
        raise ValueError(f"No sample IDs found in {split_file}")
    return sample_ids


def find_image_path(image_root, sample_id):
    image_root = Path(image_root)
    sample_path = Path(sample_id)
    if sample_path.suffix:
        candidate = image_root / sample_path
        return candidate if candidate.is_file() else None
    for suffix in IMAGE_SUFFIXES:
        candidate = image_root / f"{sample_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def find_mask_path(mask_root, sample_id):
    mask_root = Path(mask_root)
    stem = Path(sample_id).stem
    for suffix in MASK_SUFFIXES:
        candidate = mask_root / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def draw_gaussian(heatmap, center, sigma):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    # Keep floating-point ROI bounds: converting sigma * 3 to an integer first
    # changes the Gaussian support for non-integer target radii.
    temporary_size = sigma * 3
    center_col, center_row = int(center[0] + 0.5), int(center[1] + 0.5)
    height, width = heatmap.shape
    col_start = max(0, int(center_col - temporary_size))
    col_end = min(width, int(center_col + temporary_size + 1))
    row_start = max(0, int(center_row - temporary_size))
    row_end = min(height, int(center_row + temporary_size + 1))
    if col_start >= col_end or row_start >= row_end:
        return

    cols = np.arange(col_start, col_end, dtype=np.float32)
    rows = np.arange(row_start, row_end, dtype=np.float32)
    rows, cols = np.meshgrid(rows, cols, indexing="ij")
    gaussian = np.exp(
        -((cols - center_col) ** 2 + (rows - center_row) ** 2) / (2 * sigma**2)
    )
    heatmap[row_start:row_end, col_start:col_end] = np.maximum(
        heatmap[row_start:row_end, col_start:col_end],
        gaussian,
    )


def encode_mask_targets(mask, input_size=512, down_scale=4):
    """Create RPR-Net supervision directly from one binary mask."""
    if mask.ndim != 2:
        raise ValueError("Training mask must be a two-dimensional grayscale image")
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    if down_scale != 4:
        raise ValueError("RPR-Net has a fixed output stride of 4")
    if input_size % 16 != 0:
        raise ValueError("input_size must be divisible by 16")
    if input_size % down_scale != 0:
        raise ValueError("input_size must be divisible by down_scale")

    original_height, original_width = mask.shape
    output_size = input_size // down_scale
    heatmap = np.zeros((output_size, output_size), dtype=np.float32)
    radius_map = np.zeros_like(heatmap)
    offset_map = np.zeros((2, output_size, output_size), dtype=np.float32)
    regression_mask = np.zeros_like(heatmap)

    binary = (mask > 127).astype(np.uint8)
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    scale_row = input_size / original_height
    scale_col = input_size / original_width
    average_scale = (scale_row + scale_col) / 2.0

    for component_index in range(1, component_count):
        area = float(stats[component_index, cv2.CC_STAT_AREA])
        center_col, center_row = centroids[component_index]
        row_feature = center_row * scale_row / down_scale
        col_feature = center_col * scale_col / down_scale
        row_index = int(row_feature)
        col_index = int(col_feature)
        if not (0 <= row_index < output_size and 0 <= col_index < output_size):
            continue

        radius_original = np.sqrt(area / np.pi)
        radius_feature = radius_original * average_scale / down_scale
        sigma = max(2.0, min(radius_original, 4.0))
        draw_gaussian(heatmap, (col_index, row_index), sigma)
        radius_map[row_index, col_index] = radius_feature
        offset_map[0, row_index, col_index] = row_feature - row_index
        offset_map[1, row_index, col_index] = col_feature - col_index
        regression_mask[row_index, col_index] = 1.0

    return {
        "heatmap": torch.from_numpy(heatmap[None]),
        "radius_map": torch.from_numpy(radius_map[None]),
        "offset_map": torch.from_numpy(offset_map),
        "reg_mask": torch.from_numpy(regression_mask[None]),
    }


def preprocess(image_rgb):
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image.transpose(2, 0, 1)


class TemporalFrameReader:
    def __init__(self, sample_ids, image_root, input_size=512, num_frames=3):
        if num_frames not in (1, 3):
            raise ValueError("RPR-Net supports one frame or three temporal frames")
        if input_size <= 0 or input_size % 16 != 0:
            raise ValueError("input_size must be a positive multiple of 16")
        self.sample_ids = list(sample_ids)
        self.sample_id_set = set(self.sample_ids)
        self.image_root = Path(image_root)
        self.input_size = int(input_size)
        self.num_frames = int(num_frames)
        half = self.num_frames // 2
        self.offsets = tuple(range(-half, half + 1))
        if self.num_frames == 3:
            malformed = [
                sample_id
                for sample_id in self.sample_ids
                if self._parse_frame_id(sample_id) is None
            ]
            if malformed:
                preview = ", ".join(malformed[:5])
                raise ValueError(
                    "Three-frame input requires names ending in an underscore and "
                    f"a numeric frame index. Invalid examples: {preview}"
                )

    @staticmethod
    def _parse_frame_id(sample_id):
        parts = sample_id.split("_")
        if len(parts) < 2:
            return None
        try:
            frame_id = int(parts[-1])
        except ValueError:
            return None
        return "_".join(parts[:-1]), frame_id, len(parts[-1])

    def _neighbor_id(self, center_id, offset):
        if offset == 0:
            return center_id
        parsed = self._parse_frame_id(center_id)
        if parsed is None:
            return center_id
        prefix, frame_id, width = parsed
        step = 1 if offset < 0 else -1
        current_offset = offset
        while current_offset != 0:
            candidate_frame = frame_id + current_offset
            if candidate_frame >= 0:
                candidate = f"{prefix}_{candidate_frame:0{width}d}"
                if candidate in self.sample_id_set and find_image_path(self.image_root, candidate):
                    return candidate
            current_offset += step
        return center_id

    def center_path(self, sample_id):
        path = find_image_path(self.image_root, sample_id)
        if path is None:
            raise FileNotFoundError(f"Image not found for sample '{sample_id}' in {self.image_root}")
        return path

    def load_tensor(self, sample_id):
        frames = []
        for offset in self.offsets:
            neighbor_id = self._neighbor_id(sample_id, offset)
            image_path = self.center_path(neighbor_id)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Failed to read image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(
                image,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_LINEAR,
            )
            frames.append(torch.from_numpy(preprocess(image)))
        return torch.stack(frames, dim=0).float()


class RPRNetTrainDataset(Dataset):
    """Load temporal images and create supervision from binary masks."""

    def __init__(self, split_file, image_root, mask_root, input_size=512, num_frames=3, down_scale=4):
        self.sample_ids = read_sample_ids(split_file)
        self.mask_root = Path(mask_root)
        self.input_size = int(input_size)
        self.down_scale = int(down_scale)
        self.frame_reader = TemporalFrameReader(
            self.sample_ids,
            image_root=image_root,
            input_size=input_size,
            num_frames=num_frames,
        )

        missing = [
            sample_id
            for sample_id in self.sample_ids
            if find_mask_path(self.mask_root, sample_id) is None
        ]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Missing masks for {len(missing)} samples in {self.mask_root}. "
                f"Examples: {preview}"
            )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        sample_id = self.sample_ids[index]
        images = self.frame_reader.load_tensor(sample_id)
        mask_path = find_mask_path(self.mask_root, sample_id)
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for sample '{sample_id}' in {self.mask_root}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(f"Failed to read mask: {mask_path}")
        target = encode_mask_targets(
            mask,
            input_size=self.input_size,
            down_scale=self.down_scale,
        )
        return images, target, sample_id


class RPRNetPredictionDataset(Dataset):
    """Load temporal images without labels or reference masks."""

    def __init__(self, split_file, image_root, input_size=512, num_frames=3):
        self.sample_ids = read_sample_ids(split_file)
        self.frame_reader = TemporalFrameReader(
            self.sample_ids,
            image_root=image_root,
            input_size=input_size,
            num_frames=num_frames,
        )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        return self.frame_reader.load_tensor(sample_id), sample_id
