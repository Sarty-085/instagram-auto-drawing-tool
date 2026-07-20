"""test_harness.py — Offline stroke renderer & multi-algorithm visual comparison tool.

Allows testing, tuning, and comparing drawing algorithms (Legacy scanlines,
Perceptual CIELAB quantization, Directional Hatching, and Bayer Dithering)
without requiring a physical phone or ADB connection.
"""

from __future__ import annotations
import os
import sys
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

from color_mapping import COLORS_PALETTE, prepare_source_image, quantize_image
from perceptual_color import quantize_perceptual
from dithering import apply_bayer_dithering
from smart_strokes import get_directional_hatching_paths
from drawing_engine import get_fill_paths_from_mask
from metrics import compute_ssim, compute_mean_delta_e, compute_stroke_metrics


def render_strokes_canvas(
    canvas_shape: Tuple[int, int, int],
    layers_data: List[Dict[str, Any]],
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    brush_thickness: int = 8
) -> np.ndarray:
    """Simulate drawing strokes onto a digital canvas.

    Parameters
    ----------
    canvas_shape : tuple (H, W, 3)
        Dimensions of output canvas.
    layers_data : list of dict
        Each dict has: 'bgr', 'paths' (list of np.ndarray), 'mode'.
    bg_color : tuple (B, G, R)
        Background color of the canvas.
    brush_thickness : int
        Thickness of simulated brush strokes in pixels.

    Returns
    -------
    np.ndarray
        Simulated BGR drawing image.
    """
    canvas = np.full(canvas_shape, bg_color, dtype=np.uint8)

    for layer in layers_data:
        color_bgr = tuple(int(c) for c in layer["bgr"])
        paths = layer["paths"]

        for path in paths:
            if len(path) == 0:
                continue
            pts = path.reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=False, color=color_bgr, thickness=brush_thickness, lineType=cv2.LINE_AA)

    return canvas


def run_experiment_pipeline(
    img_bgra: np.ndarray,
    algorithm: str = "hatching",
    fill_step: int = 6,
    quant_metric: str = "ciede2000"
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Run full drawing simulation pipeline for a given algorithm strategy."""
    h, w = img_bgra.shape[:2]
    img_bgr = img_bgra[:, :, :3]
    fg_alpha = img_bgra[:, :, 3] if img_bgra.shape[2] == 4 else np.full((h, w), 255, dtype=np.uint8)

    # 1. Quantization & layer generation
    if algorithm == "legacy":
        _quant_bgr, closest_indices = quantize_image(img_bgr)
    elif algorithm in ("perceptual", "hatching", "cnn"):
        _quant_bgr, closest_indices, _de = quantize_perceptual(img_bgr, COLORS_PALETTE, metric=quant_metric)
    elif algorithm == "dithered":
        _quant_bgr, closest_indices = apply_bayer_dithering(img_bgr, COLORS_PALETTE, bayer_size=4, delta_e_threshold=10.0)

    # Mask background (alpha == 0) so background isn't drawn over
    closest_indices_masked = closest_indices.copy()
    closest_indices_masked[fg_alpha == 0] = -1

    unique_indices = np.unique(closest_indices_masked)
    unique_indices = unique_indices[unique_indices >= 0]

    # Sort layers lightest -> darkest by luminance
    def _luminance(idx: int) -> float:
        b, g, r = COLORS_PALETTE[idx][0]
        return 0.299 * r + 0.587 * g + 0.114 * b

    sorted_indices = sorted(unique_indices, key=_luminance, reverse=True)

    layers_data: List[Dict[str, Any]] = []
    all_paths: List[np.ndarray] = []

    for pal_idx in sorted_indices:
        mask = ((closest_indices == pal_idx) & (fg_alpha > 0)).astype(np.uint8) * 255
        if np.count_nonzero(mask) == 0:
            continue
        color_bgr = COLORS_PALETTE[pal_idx][0]

        if algorithm == "legacy":
            paths = get_fill_paths_from_mask(mask, fill_step)
        elif algorithm == "perceptual":
            paths = get_fill_paths_from_mask(mask, fill_step)
        elif algorithm == "cnn":
            paths = get_directional_hatching_paths(mask, img_bgr, step_size=fill_step, max_stroke_length=50, use_cnn=True)
        else:  # 'hatching' or 'dithered'
            paths = get_directional_hatching_paths(mask, img_bgr, step_size=fill_step, max_stroke_length=50, use_cnn=False)

        layers_data.append({
            "pal_idx": pal_idx,
            "bgr": color_bgr,
            "paths": paths
        })
        all_paths.extend(paths)

    # Detect background color from top-left corner
    bg_color = tuple(int(c) for c in img_bgr[0, 0])

    # 2. Render simulated drawing with overlapping brush thickness
    rendered_canvas = render_strokes_canvas(
        img_bgr.shape,
        layers_data,
        bg_color=bg_color,
        brush_thickness=max(6, fill_step + 3)
    )

    # 3. Calculate metrics
    ssim = compute_ssim(rendered_canvas, img_bgr)
    mean_de = compute_mean_delta_e(rendered_canvas, img_bgr)
    stroke_info = compute_stroke_metrics(all_paths)

    metrics = {
        "algorithm": algorithm,
        "ssim": round(ssim, 4),
        "mean_delta_e": round(mean_de, 2),
        "stroke_count": stroke_info["stroke_count"],
        "total_length_px": stroke_info["total_length_px"]
    }

    return rendered_canvas, layers_data, metrics


def run_visual_comparison(img_path: str) -> None:
    """Run interactive side-by-side comparison window for an image."""
    bgra = prepare_source_image(img_path)
    if bgra is None:
        print(f"Error: Could not load {img_path}")
        return

    # Resize to standard height for clear display
    max_h = 400
    h, w = bgra.shape[:2]
    scale = max_h / float(h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_bgra_resized = cv2.resize(bgra, (new_w, new_h))
    img_bgr = img_bgra_resized[:, :, :3]

    print("\n" + "=" * 60)
    print("  INSTAGRAM AUTO-DRAWING AI EXPERIMENT — TEST HARNESS")
    print("=" * 60)
    print(f"Loaded image: {img_path} ({new_w}x{new_h}px)")
    print("Running multi-algorithm experiment pipeline ...\n")

    # Run all 5 algorithms
    canvas_legacy, _, m_legacy = run_experiment_pipeline(img_bgra_resized, "legacy")
    canvas_perc, _, m_perc = run_experiment_pipeline(img_bgra_resized, "perceptual")
    canvas_hatch, _, m_hatch = run_experiment_pipeline(img_bgra_resized, "hatching")
    canvas_dither, _, m_dither = run_experiment_pipeline(img_bgra_resized, "dithered")
    canvas_cnn, _, m_cnn = run_experiment_pipeline(img_bgra_resized, "cnn")

    # Display metrics comparison table
    print("+" + "-" * 14 + "+" + "-" * 9 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 16 + "+")
    print(f"| {'Algorithm':<12} | {'SSIM ^':<7} | {'Mean dE 2000v':<12} | {'Stroke Count':<12} | {'Total Length px':<14} |")
    print("+" + "-" * 14 + "+" + "-" * 9 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 16 + "+")
    for m in [m_legacy, m_perc, m_hatch, m_dither, m_cnn]:
        print(f"| {m['algorithm']:<12} | {m['ssim']:<7.4f} | {m['mean_delta_e']:<12.2f} | {m['stroke_count']:<12d} | {m['total_length_px']:<14.1f} |")
    print("+" + "-" * 14 + "+" + "-" * 9 + "+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 16 + "+")

    # Build side-by-side 2x2 grid preview
    def _add_label(img: np.ndarray, title: str, subtitle: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), (30, 30, 30), -1)
        cv2.putText(out, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, subtitle, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        return out

    lbl_orig = _add_label(img_bgr, "Original Image", f"Reference")
    lbl_legacy = _add_label(canvas_legacy, "1. Baseline (Legacy)", f"SSIM:{m_legacy['ssim']} | dE:{m_legacy['mean_delta_e']}")
    lbl_hatch = _add_label(canvas_hatch, "2. Perceptual + Hatching", f"SSIM:{m_hatch['ssim']} | dE:{m_hatch['mean_delta_e']}")
    lbl_dither = _add_label(canvas_dither, "3. Dithered + Hatching", f"SSIM:{m_dither['ssim']} | dE:{m_dither['mean_delta_e']}")

    top_row = np.hstack([lbl_orig, lbl_legacy])
    bot_row = np.hstack([lbl_hatch, lbl_dither])
    grid = np.vstack([top_row, bot_row])

    window_name = "AI Drawing Experiment — Comparison (Press S to Save, ESC to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, grid)

    print("\nVisual comparison window open.")
    print("  Press 'S' to save output comparison grid image.")
    print("  Press 'ESC' or 'Q' to close.")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            cv2.destroyAllWindows()
            break
        elif key in (ord('s'), ord('S')):
            save_path = "experiment_comparison.png"
            cv2.imwrite(save_path, grid)
            print(f"[OK] Saved comparison grid to {save_path}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "samples/sleeping.png"
    run_visual_comparison(target)
