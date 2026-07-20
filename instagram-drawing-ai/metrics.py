"""metrics.py — Quantitative quality evaluation for drawing simulation.

Measures:
- SSIM (Structural Similarity Index) vs original image.
- Mean Delta E (perceptual CIELAB color error).
- Total stroke count and total path travel length.
"""

from __future__ import annotations
from typing import Dict, Any, List
import cv2
import numpy as np
from perceptual_color import bgr2lab_float, delta_e_ciede2000


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two BGR images."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1**2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2**2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def compute_mean_delta_e(img_bgr: np.ndarray, ref_bgr: np.ndarray) -> float:
    """Compute mean CIEDE2000 color error across all non-background pixels."""
    lab1 = bgr2lab_float(img_bgr.reshape(-1, 3))
    lab2 = bgr2lab_float(ref_bgr.reshape(-1, 3))
    de = delta_e_ciede2000(lab1, lab2)
    return float(np.mean(de))


def compute_stroke_metrics(paths_list: List[np.ndarray]) -> Dict[str, Any]:
    """Compute total stroke count and total length in pixels."""
    total_strokes = len(paths_list)
    total_length = 0.0

    for path in paths_list:
        if len(path) > 1:
            pts = path.reshape(-1, 2)
            diffs = np.diff(pts, axis=0)
            dist = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
            total_length += float(dist)

    return {
        "stroke_count": total_strokes,
        "total_length_px": round(total_length, 1)
    }
