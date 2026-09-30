"""Color palette definitions and image quantization for the Instagram auto-drawing bot.

This module provides:
- ``COLORS_PALETTE`` – the 22-colour Instagram drawing palette with screen
  coordinates for each colour swatch.
- ``remove_background`` – flood-fill-based background detection.
- ``prepare_source_image`` – loads an image and ensures it has an alpha channel.
- ``quantize_image`` – maps every pixel to the nearest palette colour using
  vectorised Euclidean distance.
"""

from __future__ import annotations

import os
from typing import Tuple, List

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Instagram drawing palette
# ---------------------------------------------------------------------------
# Each entry is (BGR_tuple, page_number, x_coordinate, color_name).
# The *page_number* identifies the palette page inside the Instagram colour
# picker, and *x_coordinate* is the horizontal screen position of the swatch.
# ---------------------------------------------------------------------------

PaletteEntry = Tuple[Tuple[int, int, int], int, int, str]

COLORS_PALETTE: List[PaletteEntry] = [
    # ---- Page 1 ----
    ((255, 255, 255), 1, 432, "white"),
    ((241, 151, 56),  1, 515, "blue"),
    ((79, 192, 112),  1, 597, "green"),
    ((65, 200, 254),  1, 680, "yellow"),
    ((51, 141, 252),  1, 763, "orange"),
    ((87, 73, 238),   1, 845, "red"),
    ((106, 7, 209),   1, 928, "pink"),
    # ---- Page 2 ----
    ((186, 7, 162),   2, 432, "purple"),
    ((20, 0, 237),    2, 515, "instagram_red"),
    ((142, 133, 237), 2, 597, "rose"),
    ((212, 211, 255), 2, 680, "light_pink"),
    ((179, 219, 254), 2, 763, "pale_orange"),
    ((130, 196, 255), 2, 845, "peach"),
    ((70, 144, 210),  2, 928, "gold_brown"),
    # ---- Page 3 ----
    ((58, 100, 153),  3, 432, "brown"),
    ((38, 38, 38),    3, 515, "black"),
    ((54, 54, 54),    3, 597, "dark_grey"),
    ((85, 85, 85),    3, 680, "grey"),
    ((115, 115, 115), 3, 763, "light_mid_grey"),
    ((153, 153, 153), 3, 845, "mid_grey"),
    ((178, 178, 178), 3, 928, "light_grey"),
    # ---- Page 4 ----
    ((199, 199, 199), 4, 432, "very_light_grey"),
]


def remove_background(img: np.ndarray) -> np.ndarray:
    """Detect background pixels using flood-fill from the four image corners.

    The function applies ``cv2.floodFill`` with fixed-range thresholding
    (lo/up diff of 15 per channel) starting from each corner.  Pixels reached
    by *any* of the four fills are considered background.

    Parameters
    ----------
    img:
        A BGR image (``np.uint8``, shape ``(H, W, 3)``).

    Returns
    -------
    np.ndarray
        A boolean mask of shape ``(H, W)`` where ``True`` marks background
        pixels.
    """
    h, w = img.shape[:2]

    # Flood-fill requires a mask that is 2 pixels larger in each dimension.
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    lo_diff: Tuple[int, int, int] = (15, 15, 15)
    up_diff: Tuple[int, int, int] = (15, 15, 15)
    fill_flags = (
        4                                  # 4-connectivity
        | (255 << 8)                       # fill value written into mask
        | cv2.FLOODFILL_FIXED_RANGE        # compare to seed, not neighbours
    )

    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]

    for cx, cy in corners:
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(
            img.copy(),
            ff_mask,
            seedPoint=(cx, cy),
            newVal=(0, 0, 0),
            loDiff=lo_diff,
            upDiff=up_diff,
            flags=fill_flags,
        )
        # The flood-fill mask has a 1-pixel border; strip it.
        combined_mask |= ff_mask[1:-1, 1:-1]

    return combined_mask.astype(bool)


def prepare_source_image(img_path: str) -> np.ndarray | None:
    """Load an image and ensure it has an alpha channel (BGRA).

    - If the image already has 4 channels (BGRA), it is returned as-is.
    - If it has 3 channels (BGR), :func:`remove_background` is called to
      generate an alpha mask (``0`` for background, ``255`` for foreground).

    Parameters
    ----------
    img_path:
        Filesystem path to the source image.

    Returns
    -------
    np.ndarray | None
        The BGRA image (``np.uint8``), or ``None`` if the file could not be
        loaded or does not exist.
    """
    if not os.path.isfile(img_path):
        print(f"[color_mapping] File not found: {img_path}")
        return None

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[color_mapping] Failed to load image: {img_path}")
        return None

    if img.ndim == 2:
        # Greyscale → convert to BGR first, then add alpha.
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    channels = img.shape[2] if img.ndim == 3 else 1

    if channels == 4:
        # Auto-heal enclosed transparent holes (e.g. white cartoon bodies that were
        # accidentally made transparent by external background removal tools).
        h, w = img.shape[:2]
        solid = (img[:, :, 3] > 50).astype(np.uint8)
        bg = solid.copy()
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        # Flood fill from all four corners to find true external background
        for corner in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            if bg[corner[1], corner[0]] == 0:
                cv2.floodFill(bg, mask, corner, 255)

        enclosed_holes = (bg == 0) & (img[:, :, 3] < 50)
        hole_count = int(np.sum(enclosed_holes))
        if hole_count > 100:
            print(f"[color_mapping] Restored {hole_count} enclosed transparent pixels to white fill.")
            img[enclosed_holes] = [255, 255, 255, 255]

        return img

    # 3-channel BGR image → create alpha from background detection.
    bg_mask = remove_background(img)
    alpha = np.where(bg_mask, 0, 255).astype(np.uint8)
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


def quantize_image(img: np.ndarray, method: str = "ciede2000") -> tuple[np.ndarray, np.ndarray]:
    """Map every pixel to the closest colour in :data:`COLORS_PALETTE`.

    Parameters
    ----------
    img:
        A BGR image (``np.uint8``, shape ``(H, W, 3)``).
    method:
        Quantization metric: ``'ciede2000'`` (perceptual Delta E), ``'cie76'``,
        or ``'rgb'`` (Euclidean).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(quantized_bgr, closest_indices)``
    """
    if method in ("ciede2000", "cie76"):
        from perceptual_color import quantize_perceptual
        quant_bgr, closest_idx, _de = quantize_perceptual(img, COLORS_PALETTE, metric=method)
        return quant_bgr, closest_idx
    else:  # RGB Euclidean
        h, w = img.shape[:2]
        palette = np.array([entry[0] for entry in COLORS_PALETTE], dtype=np.float64)
        pixels = img[:, :, :3].reshape(-1, 3).astype(np.float64)
        pixel_sq = np.sum(pixels ** 2, axis=1, keepdims=True)
        palette_sq = np.sum(palette ** 2, axis=1, keepdims=True)
        dists = pixel_sq + palette_sq.T - 2.0 * pixels @ palette.T
        closest = np.argmin(dists, axis=1).astype(np.int32)
        quantized_flat = palette[closest].astype(np.uint8)
        return quantized_flat.reshape(h, w, 3), closest.reshape(h, w)


def extend_palette_from_config(config: dict) -> None:
    """Dynamically append custom spectrum colours to global COLORS_PALETTE if calibrated."""
    spectrum_cfg = config.get("spectrum")
    if not spectrum_cfg or "colors" not in spectrum_cfg:
        return

    # Check if already extended (to avoid duplicate appends on repeated calls)
    if len(COLORS_PALETTE) > 22:
        # Reset back to base 22 colors first
        del COLORS_PALETTE[22:]

    colors = spectrum_cfg["colors"]
    start_x = spectrum_cfg["start_x"]
    
    for idx, bgr in enumerate(colors):
        x_coord = start_x + idx
        # Format: (BGR_tuple, page_number, x_coordinate, color_name)
        COLORS_PALETTE.append(
            (tuple(bgr), -1, x_coord, f"custom_{idx}")
        )
    print(f"Dynamic palette extended with {len(colors)} custom spectrum colours.")


def clean_antialiasing_halos(
    closest_indices: np.ndarray,
    fg_bgr: np.ndarray,
    fg_alpha: np.ndarray,
    min_core_pixels: int = 0,
) -> np.ndarray:
    """Detect and eliminate anti-aliasing edge halos (e.g. spurious greys).

    In digital artwork and doodles, outlines have 1-2 pixel transition
    gradients (anti-aliasing) that quantize to multiple shades of grey,
    dark grey, or intermediate hues. These halos have virtually no solid
    interior core and should not count as separate drawing layers.

    This function identifies genuine layers whose masks retain >= min_core_pixels
    after a 1-pixel morphological erosion, and snaps all halo pixels to the
    nearest genuine layer by colour distance.
    """
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = fg_alpha > 0
    total_fg = int(np.sum(fg_mask))
    if min_core_pixels <= 0:
        min_core_pixels = max(80, int(total_fg * 0.003))

    unique_indices = np.unique(closest_indices[fg_mask])

    genuine_layers = []
    for idx in unique_indices:
        mask = ((closest_indices == idx) & fg_mask).astype(np.uint8)
        eroded = cv2.erode(mask, kernel)
        if np.sum(eroded) >= min_core_pixels:
            genuine_layers.append(int(idx))

    if not genuine_layers:
        return closest_indices

    cleaned = closest_indices.copy()
    halo_mask = ~np.isin(closest_indices, genuine_layers) & fg_mask
    if not np.any(halo_mask):
        return cleaned

    gen_colors = np.array([COLORS_PALETTE[i][0] for i in genuine_layers], dtype=np.float64)
    halo_pixels = fg_bgr[halo_mask].astype(np.float64)
    dists = np.sum((halo_pixels[:, None, :] - gen_colors[None, :, :]) ** 2, axis=2)
    best_gen_idx = np.argmin(dists, axis=1)
    cleaned[halo_mask] = np.array(genuine_layers)[best_gen_idx]

    halo_count = np.sum(halo_mask)
    if halo_count > 0:
        print(f"[color_mapping] Merged {halo_count} anti-aliased halo pixels into dominant layers.")

    return cleaned

