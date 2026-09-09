"""Decode RPR-Net outputs and render radius or PAMG masks."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from labelSIRST.pamg import pamg_grow_mask_fast


def _local_maximum(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    pooled = F.max_pool2d(heatmap, kernel, stride=1, padding=padding)
    return heatmap * pooled.eq(heatmap).to(heatmap.dtype)


def decode_centers(heatmap, offset_map, topk=100, score_threshold=0.1):
    """Return decoded centers and their source heatmap cells for radius lookup."""
    if heatmap.ndim != 4 or heatmap.shape[1] != 1:
        raise ValueError("heatmap must have shape [B, 1, H, W]")
    if offset_map.ndim != 4 or offset_map.shape[1] != 2:
        raise ValueError("offset_map must have shape [B, 2, H, W]")
    if heatmap.shape[0] != offset_map.shape[0] or heatmap.shape[2:] != offset_map.shape[2:]:
        raise ValueError("heatmap and offset_map batch/spatial dimensions must match")
    if int(topk) < 1:
        raise ValueError("topk must be at least 1")
    if not 0.0 <= float(score_threshold) <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1")

    heatmap = _local_maximum(heatmap)
    batch, _, height, width = heatmap.shape
    count = min(int(topk), height * width)
    scores, indices = torch.topk(heatmap.view(batch, -1), count)

    row_indices = indices // width
    col_indices = indices % width
    rows = row_indices.float()
    cols = col_indices.float()
    offsets = offset_map.view(batch, 2, -1).gather(
        2, indices.unsqueeze(1).expand(-1, 2, -1)
    )
    rows = rows + offsets[:, 0]
    cols = cols + offsets[:, 1]

    decoded = []
    for batch_index in range(batch):
        detections = []
        for item_index in range(count):
            score = float(scores[batch_index, item_index].item())
            if score < score_threshold:
                continue
            detections.append({
                "row_feature": float(rows[batch_index, item_index].item()),
                "col_feature": float(cols[batch_index, item_index].item()),
                "row_index": int(row_indices[batch_index, item_index].item()),
                "col_index": int(col_indices[batch_index, item_index].item()),
                "score": score,
            })
        decoded.append(detections)
    return decoded


def _input_hw(input_size):
    if isinstance(input_size, int):
        return input_size, input_size
    return int(input_size[0]), int(input_size[1])


def point_to_original(row_feature, col_feature, original_hw, input_size=512, down_scale=4):
    input_height, input_width = _input_hw(input_size)
    original_height, original_width = original_hw
    row = row_feature * down_scale * original_height / input_height
    col = col_feature * down_scale * original_width / input_width
    row = float(np.clip(row, 0, original_height - 1))
    col = float(np.clip(col, 0, original_width - 1))
    return row, col


def radius_to_original(radius_feature, original_hw, input_size=512, down_scale=4):
    input_height, input_width = _input_hw(input_size)
    original_height, original_width = original_hw
    scale_height = input_height / original_height
    scale_width = input_width / original_width
    average_scale = (scale_height + scale_width) / 2.0
    return float(radius_feature * down_scale / average_scale)


def detection_records(points, radius_map, batch_index, original_hw, input_size=512, down_scale=4):
    if radius_map.ndim != 4 or radius_map.shape[1] != 1:
        raise ValueError("radius_map must have shape [B, 1, H, W]")
    if not 0 <= batch_index < radius_map.shape[0]:
        raise IndexError("batch_index is outside radius_map")
    _, _, feature_height, feature_width = radius_map.shape
    records = []
    for point in points:
        row_feature = point["row_feature"]
        col_feature = point["col_feature"]
        score = point["score"]
        row_index = int(np.clip(point["row_index"], 0, feature_height - 1))
        col_index = int(np.clip(point["col_index"], 0, feature_width - 1))
        radius_feature = float(radius_map[batch_index, 0, row_index, col_index].item())
        row, col = point_to_original(
            row_feature,
            col_feature,
            original_hw,
            input_size=input_size,
            down_scale=down_scale,
        )
        radius = max(
            1.0,
            radius_to_original(
                radius_feature,
                original_hw,
                input_size=input_size,
                down_scale=down_scale,
            ),
        )
        records.append({"row": row, "col": col, "radius": radius, "score": score})
    return records


def render_radius_mask(records, original_hw):
    mask = np.zeros(original_hw, dtype=np.uint8)
    for record in records:
        center = (int(round(record["col"])), int(round(record["row"])))
        radius = max(1, int(round(record["radius"])))
        cv2.circle(mask, center, radius, color=255, thickness=-1)
    return mask


def render_pamg_mask(records, image_gray, Rs_scale=1.0, Rs_cap=None, polarity=0):
    if image_gray.ndim != 2:
        raise ValueError("PAMG expects a grayscale image")
    mask = np.zeros_like(image_gray, dtype=np.uint8)
    for record in records:
        row = int(round(record["row"]))
        col = int(round(record["col"]))
        Rs = max(1.0, record["radius"] * float(Rs_scale))
        if Rs_cap is not None:
            Rs = min(Rs, float(Rs_cap))
        grown = pamg_grow_mask_fast(
            seed_point=(row, col),
            image=image_gray,
            Rs=max(1, int(round(Rs))),
            mode=int(polarity),
        )
        mask = np.maximum(mask, grown)
    return mask
