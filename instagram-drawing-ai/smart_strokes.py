"""smart_strokes.py — Gradient-aware directional hatching for organic stroke generation.

Replaces uniform horizontal scanlines with structure-aligned polylines that follow
image contours, producing a natural hand-drawn appearance.
"""

from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np


def compute_gradient_orientation_field(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute local edge/contour orientation angles and gradient magnitude.

    Orientation is perpendicular to the gradient direction (tangent to edges).

    Returns
    -------
    tuple (tangent_angle_rad, magnitude)
        - tangent_angle_rad: (H, W) float32 angle in radians [0, pi).
        - magnitude: (H, W) float32 gradient magnitude.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Compute Sobel gradients
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx**2 + gy**2)

    # Gradient angle is arctan2(gy, gx).
    # Tangent angle is perpendicular (+ pi/2)
    grad_angle = np.arctan2(gy, gx)
    tangent_angle = (grad_angle + np.pi / 2.0) % np.pi

    return tangent_angle.astype(np.float32), magnitude.astype(np.float32)


def get_directional_hatching_paths(
    mask: np.ndarray,
    img_bgr: np.ndarray,
    step_size: int = 6,
    max_stroke_length: int = 60,
    min_stroke_length: int = 10,
    use_cnn: bool = False
) -> List[np.ndarray]:
    """Generate directional hatching strokes aligned with gradients or CNN predictions.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (uint8, 0 or 255) for the layer.
    img_bgr : np.ndarray
        Full BGR image for orientation field computation.
    step_size : int
        Grid sampling step for seed points.
    max_stroke_length : int
        Maximum path length in pixels for a single stroke.
    min_stroke_length : int
        Minimum path length to keep stroke.
    use_cnn : bool
        If True, use Phase 5 Tiny CNN to predict stroke directions instead of Sobel gradients.

    Returns
    -------
    list[np.ndarray]
        List of stroke paths, each an int32 array of shape (N, 1, 2).
    """
    h_mask, w_mask = mask.shape[:2]
    if use_cnn:
        from cnn_strokes import compute_cnn_orientation_field
        tangent_angle, magnitude = compute_cnn_orientation_field(img_bgr)
    else:
        tangent_angle, magnitude = compute_gradient_orientation_field(img_bgr)

    # Distance transform to prioritize seed points inside thick areas
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    paths: List[np.ndarray] = []
    visited = np.zeros((h_mask, w_mask), dtype=bool)

    # Create grid of candidate seed points
    y_seeds, x_seeds = np.where((mask > 0) & (dist_transform > 1.0))
    if len(y_seeds) == 0:
        return []

    # Sort seeds by distance transform descending (inside out)
    seed_dists = dist_transform[y_seeds, x_seeds]
    sort_order = np.argsort(-seed_dists)
    y_seeds = y_seeds[sort_order]
    x_seeds = x_seeds[sort_order]

    # Sample seeds every step_size pixels
    step_mask = (y_seeds % step_size == 0) & (x_seeds % step_size == 0)
    y_seeds = y_seeds[step_mask]
    x_seeds = x_seeds[step_mask]

    integration_step = 2.0  # pixel step per streamline integration

    for sy, sx in zip(y_seeds, x_seeds):
        if visited[sy, sx]:
            continue

        # Trace streamline in both directions (+dir and -dir)
        points_forward: List[Tuple[float, float]] = [(float(sx), float(sy))]
        points_backward: List[Tuple[float, float]] = []

        # Forward trace
        cx, cy = float(sx), float(sy)
        for _ in range(max_stroke_length // 2):
            ix, iy = int(round(cx)), int(round(cy))
            if ix < 0 or ix >= w_mask or iy < 0 or iy >= h_mask or mask[iy, ix] == 0:
                break
            visited[iy, ix] = True

            angle = tangent_angle[iy, ix]
            dx = np.cos(angle) * integration_step
            dy = np.sin(angle) * integration_step

            # Prevent sharp 180 flips by checking dot product with previous step
            if len(points_forward) > 1:
                prev_dx = points_forward[-1][0] - points_forward[-2][0]
                prev_dy = points_forward[-1][1] - points_forward[-2][1]
                if prev_dx * dx + prev_dy * dy < 0:
                    dx, dy = -dx, -dy

            cx += dx
            cy += dy
            points_forward.append((cx, cy))

        # Backward trace
        cx, cy = float(sx), float(sy)
        for _ in range(max_stroke_length // 2):
            ix, iy = int(round(cx)), int(round(cy))
            if ix < 0 or ix >= w_mask or iy < 0 or iy >= h_mask or mask[iy, ix] == 0:
                break
            visited[iy, ix] = True

            angle = tangent_angle[iy, ix]
            dx = -np.cos(angle) * integration_step
            dy = -np.sin(angle) * integration_step

            if len(points_backward) > 0:
                prev_dx = points_backward[-1][0] - (points_backward[-2][0] if len(points_backward) > 1 else float(sx))
                prev_dy = points_backward[-1][1] - (points_backward[-2][1] if len(points_backward) > 1 else float(sy))
                if prev_dx * dx + prev_dy * dy < 0:
                    dx, dy = -dx, -dy

            cx += dx
            cy += dy
            points_backward.append((cx, cy))

        # Combine backward (reversed) + forward
        full_pts = points_backward[::-1] + points_forward
        if len(full_pts) * integration_step >= min_stroke_length:
            # Convert to OpenCV polyline format (N, 1, 2) int32
            pts_array = np.array(full_pts, dtype=np.int32).reshape(-1, 1, 2)
            paths.append(pts_array)

    return paths
