"""PAMG point-to-mask region growing."""

import heapq

import numpy as np


def pamg_grow_mask_fast(seed_point, image, Rs=10, mode=0):
    """Grow one infrared target mask from a point and the paper's Rs prior.

    Args:
        seed_point: ``(row, col)`` in original-image coordinates.
        image: Grayscale ``uint8`` image.
        Rs: PAMG growth-scale prior that bounds the search area.
        mode: ``1`` for bright targets, ``-1`` for dark targets, and ``0``
            for automatic polarity selection.
    """
    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale array")
    if Rs < 1:
        raise ValueError("Rs must be at least 1")

    height, width = image.shape
    seed_row, seed_col = int(seed_point[0]), int(seed_point[1])
    if not (0 <= seed_row < height and 0 <= seed_col < width):
        raise ValueError("seed_point is outside the image")

    image_float = image.astype(np.float32) / 255.0
    seed_value = image_float[seed_row, seed_col]

    if mode == 1:
        local_background = -1.0
    elif mode == -1:
        local_background = 99999.0
    elif mode == 0:
        row_start = max(0, seed_row - Rs)
        row_end = min(height, seed_row + Rs)
        col_start = max(0, seed_col - Rs)
        col_end = min(width, seed_col + Rs)
        local_background = np.median(image_float[row_start:row_end, col_start:col_end])
    else:
        raise ValueError("mode must be -1, 0, or 1")

    if seed_value < local_background:
        work_image = 1.0 - image_float
        seed_value = 1.0 - seed_value
    else:
        work_image = image_float

    priority_queue = [(-seed_value, seed_row, seed_col)]
    visited = np.zeros((height, width), dtype=bool)
    visited[seed_row, seed_col] = True
    flooded_rows = []
    flooded_cols = []

    # The original PAMG formulation intentionally applies a double seed weight.
    sum_value = seed_value
    sum_squared = seed_value**2
    interior_count = 1
    current_max_distance_squared = 0.0
    hard_area_limit = int(3.14 * Rs**2)
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))

    peak_energy = -np.inf
    peak_index = 0
    peak_distance_squared = 0.0
    iteration = 0

    while priority_queue and interior_count < hard_area_limit:
        negative_value, row, col = heapq.heappop(priority_queue)
        current_value = -negative_value
        flooded_rows.append(row)
        flooded_cols.append(col)
        interior_count += 1
        sum_value += current_value
        sum_squared += current_value**2

        distance_squared = (row - seed_row) ** 2 + (col - seed_col) ** 2
        current_max_distance_squared = max(current_max_distance_squared, distance_squared)

        for row_offset, col_offset in neighbors:
            next_row, next_col = row + row_offset, col + col_offset
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and not visited[next_row, next_col]
            ):
                visited[next_row, next_col] = True
                heapq.heappush(
                    priority_queue,
                    (-work_image[next_row, next_col], next_row, next_col),
                )

        if interior_count > 5:
            mean_inside = sum_value / interior_count
            mean_outside = -priority_queue[0][0] if priority_queue else mean_inside
            contrast = max(0.0, mean_inside - mean_outside)
            variance_inside = (sum_squared / interior_count) - mean_inside**2
            standard_deviation = np.sqrt(max(1e-8, variance_inside))
            signal_term = contrast / (standard_deviation + 0.05)
            energy = (
                np.log(np.log(interior_count))
                + np.log(signal_term + 1e-6)
                - current_max_distance_squared / (2 * Rs**2)
            )
            radius_squared = current_max_distance_squared
        else:
            energy = -999.0
            radius_squared = 0.0

        if energy > peak_energy:
            peak_energy = energy
            peak_index = iteration
            peak_distance_squared = radius_squared
        iteration += 1

    if iteration == 0 or peak_distance_squared > Rs**2:
        return np.zeros((height, width), dtype=np.uint8)

    cutoff_count = max(1, peak_index - 1)
    if not flooded_rows:
        return np.zeros((height, width), dtype=np.uint8)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[flooded_rows[:cutoff_count], flooded_cols[:cutoff_count]] = 255
    return mask


# Historical function name retained for users of the original label tool.
seed_grow_mask_unified_fast = pamg_grow_mask_fast
