"""perceptual_color.py — CIELAB & CIEDE2000 perceptual color distance and quantization.

Replaces standard BGR Euclidean distance with human-perception-based CIELAB / CIEDE2000
color distance metrics for superior palette mapping.
"""

from __future__ import annotations
from typing import List, Tuple, Union
import cv2
import numpy as np


def bgr2lab_float(bgr_array: np.ndarray) -> np.ndarray:
    """Convert a uint8 BGR image or array of BGR colors to standard floating-point CIELAB.

    L* in [0, 100], a* in [-128, 127], b* in [-128, 127].
    """
    if bgr_array.ndim == 1:
        bgr_3d = bgr_array.reshape(1, 1, 3).astype(np.uint8)
    elif bgr_array.ndim == 2:
        bgr_3d = bgr_array.reshape(-1, 1, 3).astype(np.uint8)
    else:
        bgr_3d = bgr_array.astype(np.uint8)

    lab_u8 = cv2.cvtColor(bgr_3d, cv2.COLOR_BGR2Lab).astype(np.float64)
    # OpenCV maps L -> L*255/100, a -> a+128, b -> b+128
    L = lab_u8[:, :, 0] * (100.0 / 255.0)
    a = lab_u8[:, :, 1] - 128.0
    b = lab_u8[:, :, 2] - 128.0

    lab_float = np.stack([L, a, b], axis=-1)

    if bgr_array.ndim == 1:
        return lab_float.reshape(3)
    elif bgr_array.ndim == 2:
        return lab_float.reshape(-1, 3)
    return lab_float


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Compute CIE76 color difference (Euclidean distance in CIELAB space).

    Supports broadcasting between (M, 3) and (N, 3) -> (M, N).
    """
    if lab1.ndim == 2 and lab2.ndim == 2:
        # lab1: (M, 3), lab2: (N, 3)
        diff = lab1[:, np.newaxis, :] - lab2[np.newaxis, :, :]  # (M, N, 3)
        return np.sqrt(np.sum(diff ** 2, axis=-1))
    diff = lab1 - lab2
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> np.ndarray:
    """Compute CIEDE2000 color difference (ΔE 2000) between CIELAB colors.

    lab1: (M, 3) or (H, W, 3)
    lab2: (N, 3) or same shape as lab1 for 1-to-1 comparison.
    Returns: (M, N) matrix or 1D/2D array of perceptual distances.
    """
    if lab1.shape == lab2.shape:
        L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    elif lab1.ndim == 2 and lab2.ndim == 2:
        L1 = lab1[:, 0:1]  # (M, 1)
        a1 = lab1[:, 1:2]
        b1 = lab1[:, 2:3]
        L2 = lab2[:, 0].T  # (1, N)
        a2 = lab2[:, 1].T
        b2 = lab2[:, 2].T
    elif lab1.ndim == 3 and lab2.ndim == 2:
        M = lab1.shape[0] * lab1.shape[1]
        lab1_flat = lab1.reshape(M, 3)
        res = delta_e_ciede2000(lab1_flat, lab2, kL, kC, kH)
        return res.reshape(lab1.shape[0], lab1.shape[1], lab2.shape[0])
    else:
        L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0

    G = 0.5 * (1.0 - np.sqrt(C_bar**7 / (C_bar**7 + 25.0**7 + 1e-12)))

    a1_prime = (1.0 + G) * a1
    a2_prime = (1.0 + G) * a2

    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0

    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    dh_prime = np.zeros_like(dL_prime)
    prod_C = C1_prime * C2_prime
    nonzero_mask = prod_C > 1e-8

    diff_h = h2_prime - h1_prime
    abs_diff_h = np.abs(diff_h)

    dh_prime = np.where(nonzero_mask, np.where(abs_diff_h <= 180.0, diff_h,
                       np.where(h2_prime <= h1_prime, diff_h + 360.0, diff_h - 360.0)), 0.0)

    dH_prime = 2.0 * np.sqrt(np.maximum(0.0, prod_C)) * np.sin(np.radians(dh_prime / 2.0))

    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0

    h_bar_prime = np.zeros_like(dL_prime)
    h_sum = h1_prime + h2_prime
    h_bar_prime = np.where(nonzero_mask,
                           np.where(abs_diff_h <= 180.0, h_sum / 2.0,
                           np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0)),
                           h_sum)

    T = (1.0 - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
         + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0)))

    dTheta = 30.0 * np.exp(-((h_bar_prime - 275.0) / 25.0)**2)
    R_C = 2.0 * np.sqrt(np.maximum(0.0, C_bar_prime**7 / (C_bar_prime**7 + 25.0**7 + 1e-12)))
    S_L = 1.0 + (0.015 * (L_bar_prime - 50.0)**2) / np.sqrt(np.maximum(1e-6, 20.0 + (L_bar_prime - 50.0)**2))
    S_C = 1.0 + 0.045 * C_bar_prime
    S_H = 1.0 + 0.015 * C_bar_prime * T
    R_T = -np.sin(np.radians(2.0 * dTheta)) * R_C

    term_L = dL_prime / (kL * S_L)
    term_C = dC_prime / (kC * S_C)
    term_H = dH_prime / (kH * S_H)

    de2000 = np.sqrt(np.maximum(0.0, term_L**2 + term_C**2 + term_H**2 + R_T * term_C * term_H))
    return de2000


def quantize_perceptual(img_bgr: np.ndarray, palette_entries: List[Tuple[Tuple[int, int, int], int, int, str]], metric: str = "ciede2000") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize BGR image using perceptual CIELAB color metrics.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input BGR image uint8 (H, W, 3).
    palette_entries : list of palette tuples
        Each tuple has ((B, G, R), page, x, name).
    metric : str
        'ciede2000' or 'cie76' or 'rgb'.

    Returns
    -------
    tuple (quantized_bgr, closest_indices, min_delta_e)
        - quantized_bgr: (H, W, 3) uint8 image.
        - closest_indices: (H, W) int32 index into palette.
        - min_delta_e: (H, W) float64 minimum Delta E value per pixel.
    """
    h, w = img_bgr.shape[:2]
    palette_bgr = np.array([entry[0] for entry in palette_entries], dtype=np.uint8)

    if metric == "rgb":
        pixels = img_bgr.reshape(-1, 3).astype(np.float64)
        pal = palette_bgr.astype(np.float64)
        dists = np.sqrt(np.sum((pixels[:, np.newaxis, :] - pal[np.newaxis, :, :]) ** 2, axis=-1))
    else:
        pixels_lab = bgr2lab_float(img_bgr.reshape(-1, 3))
        palette_lab = bgr2lab_float(palette_bgr)

        if metric == "cie76":
            dists = delta_e_cie76(pixels_lab, palette_lab)
        else:
            dists = delta_e_ciede2000(pixels_lab, palette_lab)

    closest = np.argmin(dists, axis=1).astype(np.int32)
    min_dists = np.min(dists, axis=1)

    quantized_flat = palette_bgr[closest]
    quantized_bgr = quantized_flat.reshape(h, w, 3)
    closest_indices = closest.reshape(h, w)
    min_delta_e = min_dists.reshape(h, w)

    return quantized_bgr, closest_indices, min_delta_e
