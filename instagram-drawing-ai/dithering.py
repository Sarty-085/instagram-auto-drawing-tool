"""dithering.py — Ordered Bayer dithering for color expansion.

Simulates intermediate shades and smooth color transitions using Instagram's
limited color palette by applying ordered Bayer matrix dithering.
"""

from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np
from perceptual_color import bgr2lab_float, delta_e_ciede2000, delta_e_cie76

# Standard 4x4 Bayer Matrix normalized to [0, 1)
BAYER_MATRIX_4X4 = np.array([
    [ 0/16,  8/16,  2/16, 10/16],
    [12/16,  4/16, 14/16,  6/16],
    [ 3/16, 11/16,  1/16,  9/16],
    [15/16,  7/16, 13/16,  5/16]
], dtype=np.float64)

# 8x8 Bayer Matrix
BAYER_MATRIX_8X8 = np.array([
    [ 0/64, 32/64,  8/64, 40/64,  2/64, 34/64, 10/64, 42/64],
    [48/64, 16/64, 56/64, 24/64, 50/64, 18/64, 58/64, 26/64],
    [12/64, 44/64,  4/64, 36/64, 14/64, 46/64,  6/64, 38/64],
    [60/64, 28/64, 52/64, 20/64, 62/64, 30/64, 54/64, 22/64],
    [ 3/64, 35/64, 11/64, 43/64,  1/64, 33/64,  9/64, 41/64],
    [51/64, 19/64, 59/64, 27/64, 49/64, 17/64, 57/64, 25/64],
    [15/64, 47/64,  7/64, 39/64, 13/64, 45/64,  5/64, 37/64],
    [63/64, 31/64, 55/64, 23/64, 61/64, 29/64, 53/64, 21/64]
], dtype=np.float64)


def apply_bayer_dithering(
    img_bgr: np.ndarray,
    palette_entries: List[Tuple[Tuple[int, int, int], int, int, str]],
    bayer_size: int = 4,
    delta_e_threshold: float = 12.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply ordered Bayer dithering to approximate intermediate colors.

    Parameters
    ----------
    img_bgr : np.ndarray
        Source BGR image (H, W, 3).
    palette_entries : list of palette tuples
        Instagram palette entries.
    bayer_size : int
        4 or 8 for Bayer matrix dimension.
    delta_e_threshold : float
        Only apply dithering to pixels whose best palette match Delta E exceeds this.

    Returns
    -------
    tuple (dithered_bgr, closest_indices)
        - dithered_bgr: (H, W, 3) BGR image after Bayer dithering.
        - closest_indices: (H, W) int32 matrix mapping pixels to palette indices.
    """
    h, w = img_bgr.shape[:2]
    palette_bgr = np.array([entry[0] for entry in palette_entries], dtype=np.uint8)
    N_pal = len(palette_bgr)

    # 1. Convert to CIELAB space
    pixels_lab = bgr2lab_float(img_bgr.reshape(-1, 3))
    palette_lab = bgr2lab_float(palette_bgr)

    # 2. Compute distances to all palette colors
    dists = delta_e_ciede2000(pixels_lab, palette_lab)  # (H*W, N_pal)

    # 3. Find top 2 closest palette indices per pixel
    sorted_idx = np.argsort(dists, axis=1)
    best_idx = sorted_idx[:, 0]
    second_idx = sorted_idx[:, 1]

    d1 = dists[np.arange(len(dists)), best_idx]
    d2 = dists[np.arange(len(dists)), second_idx]

    # Calculate mixing ratio t in [0, 1] between best (d1) and second (d2)
    # If d1 is 0, t = 0 (100% best). If d1 == d2, t = 0.5.
    denom = d1 + d2 + 1e-6
    ratio = d1 / denom  # 0 -> exact match to best, ~0.5 -> midway

    # 4. Generate tiled Bayer threshold map
    bayer_mat = BAYER_MATRIX_4X4 if bayer_size == 4 else BAYER_MATRIX_8X8
    bayer_h, bayer_w = bayer_mat.shape
    tiles_y = (h + bayer_h - 1) // bayer_h
    tiles_x = (w + bayer_w - 1) // bayer_w

    tiled_bayer = np.tile(bayer_mat, (tiles_y, tiles_x))[:h, :w].reshape(-1)

    # 5. Dither decision: if error > threshold and ratio > bayer threshold, use second_idx
    dither_mask = (d1 > delta_e_threshold) & (ratio > (tiled_bayer * 0.5))
    chosen_idx = np.where(dither_mask, second_idx, best_idx).astype(np.int32)

    dithered_flat = palette_bgr[chosen_idx]
    dithered_bgr = dithered_flat.reshape(h, w, 3)
    closest_indices = chosen_idx.reshape(h, w)

    return dithered_bgr, closest_indices
