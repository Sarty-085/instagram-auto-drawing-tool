"""draw_interactive — Main entry point for the Instagram auto-drawing bot.

This module is a thin orchestrator that wires together:

* :mod:`adb_utils` — ADB device communication
* :mod:`config` — configuration loading / saving
* :mod:`calibration` — first-run device calibration
* :mod:`color_mapping` — palette quantisation and background removal
* :mod:`gui` — interactive overlay editor and mapping dashboard
* :mod:`drawing_engine` — swipe-based drawing execution

Run from the command line::

    python draw_interactive.py [image_path] [--recalibrate]
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

from adb_utils import ADBConnection
from calibration import DeviceCalibration
from color_mapping import COLORS_PALETTE, prepare_source_image, quantize_image, extend_palette_from_config
from config import load_config, get_config_path
from drawing_engine import execute_drawing
from gui import overlay_image_bgra, run_overlay_editor, run_mapping_dashboard, run_eraser_editor


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

MIN_LAYER_FRACTION = 0.015   # Layers below 1.5% of foreground pixels are noise
MIN_LAYER_PIXELS_ABS = 100   # Absolute floor for tiny images
MIN_CONTOUR_AREA = 15.0      # Contours smaller than this are ignored
AUTO_FILL_THRESHOLD = 3000   # Layers > this many pixels default to FILL mode


def _resolve_image_path(cli_arg: str | None) -> str | None:
    """Prompt or resolve the source image path.

    Parameters
    ----------
    cli_arg : str or None
        Path passed on the command line, if any.

    Returns
    -------
    str or None
        Resolved absolute path, or ``None`` if the file was not found.
    """
    if cli_arg:
        path = cli_arg
    else:
        path = input(
            "Enter path to the image to draw (default: samples/sleeping.png): "
        ).strip()
        if not path:
            path = "samples/sleeping.png"
            # Fallback: check old location if samples/ doesn't exist
            if not os.path.exists(path) and os.path.exists("sleeping.png"):
                path = "sleeping.png"

    if not os.path.exists(path):
        print(f"Error: File '{path}' not found.")
        return None
    return path


def main() -> None:
    """Main application flow."""
    # ------------------------------------------------------------------
    # 0. Parse simple CLI arguments
    # ------------------------------------------------------------------
    image_arg = None
    recalibrate = False
    for arg in sys.argv[1:]:
        if arg in ("--recalibrate", "-r"):
            recalibrate = True
        else:
            image_arg = arg

    # ------------------------------------------------------------------
    # 1. Connect to device
    # ------------------------------------------------------------------
    try:
        adb = ADBConnection()
    except (FileNotFoundError, ConnectionError) as exc:
        print(f"\n❌ {exc}")
        print("\nTroubleshooting:")
        print("  → Is USB Debugging enabled? (Settings > Developer Options)")
        print("  → Is your phone connected via USB?")
        print("  → Try running: adb devices")
        return

    # ------------------------------------------------------------------
    # 2. Load or create configuration
    # ------------------------------------------------------------------
    config_path = get_config_path()
    config = load_config(config_path)

    if recalibrate or not os.path.exists(config_path):
        print("\n[CALIBRATION] Running device calibration...")
        calibrator = DeviceCalibration(adb)
        config = calibrator.run_calibration()
    else:
        print(f"Config loaded from {config_path}")

    # Extend palette with custom spectrum colors if calibrated
    extend_palette_from_config(config)

    # ------------------------------------------------------------------
    # 3. Load and prepare source image
    # ------------------------------------------------------------------
    image_path = _resolve_image_path(image_arg)
    if image_path is None:
        return

    source_bgra = prepare_source_image(image_path)
    if source_bgra is None:
        print("Error: Could not load or prepare image.")
        return

    print(f"Source image loaded: {image_path}")

    # ------------------------------------------------------------------
    # 4. Capture phone screenshot
    # ------------------------------------------------------------------
    print("Capturing phone screen...")
    try:
        screenshot = adb.screenshot()
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return

    # ------------------------------------------------------------------
    # 5. Interactive overlay placement
    # ------------------------------------------------------------------
    result = run_overlay_editor(screenshot, source_bgra)
    if result is None:
        return

    ov_x, ov_y, ov_w, ov_h, scale_factor = result

    # Translate to full phone-screen coordinates
    x_phone = int(ov_x / scale_factor)
    y_phone = int(ov_y / scale_factor)
    w_phone = int(ov_w / scale_factor)
    h_phone = int(ov_h / scale_factor)

    print(f"\n  Phone coordinates: X={x_phone}, Y={y_phone}, "
          f"W={w_phone}, H={h_phone}")

    # ------------------------------------------------------------------
    # 6. Quantise image to Instagram palette
    # ------------------------------------------------------------------
    resized_fg = cv2.resize(source_bgra, (w_phone, h_phone))
    fg_bgr = cv2.medianBlur(resized_fg[:, :, :3], 3)
    fg_alpha = resized_fg[:, :, 3]

    print("Quantising colours to Instagram palette...")
    _quantized_bgr, closest_indices_img = quantize_image(fg_bgr)

    # ------------------------------------------------------------------
    # 7. Identify active layers (filter noise)
    # ------------------------------------------------------------------
    unique_indices, counts = np.unique(
        closest_indices_img[fg_alpha > 0], return_counts=True
    )
    total_fg = int(np.sum(fg_alpha > 0))
    min_pixels = max(MIN_LAYER_PIXELS_ABS, int(total_fg * MIN_LAYER_FRACTION))

    sorted_pairs = sorted(zip(unique_indices, counts), key=lambda p: -p[1])
    present_indices = [int(idx) for idx, cnt in sorted_pairs if cnt >= min_pixels]

    if not present_indices:
        print("Error: No visible layers detected in the foreground.")
        return

    # Default layer modes
    layer_modes = {}
    for idx, cnt in sorted_pairs:
        layer_modes[int(idx)] = "fill" if cnt > AUTO_FILL_THRESHOLD else "outline"

    # Re-order layers lightest → darkest for correct painting order.
    # In cartoon art: light fills first (white pillow, coloured body),
    # dark outlines drawn last so they appear on top.
    def _luminance(pal_idx: int) -> float:
        bgr = COLORS_PALETTE[pal_idx][0]
        b, g, r = bgr
        return 0.299 * r + 0.587 * g + 0.114 * b

    present_indices.sort(key=_luminance, reverse=True)  # brightest first

    print(f"  {len(present_indices)} colour layers detected.")


    # ------------------------------------------------------------------
    # 8. Mapping dashboard
    # ------------------------------------------------------------------
    dashboard_result = run_mapping_dashboard(
        closest_indices_img, fg_alpha, present_indices, layer_modes, config
    )
    if dashboard_result is None:
        return

    final_present = dashboard_result["present_color_indices"]
    final_mapped = dashboard_result["mapped_color_indices"]
    final_modes = dashboard_result["layer_modes"]
    bg_color_idx = dashboard_result["bg_color_idx"]

    # ------------------------------------------------------------------
    # 9. Build layers data for drawing engine
    # ------------------------------------------------------------------
    layers_data = []
    for orig_idx in final_present:
        mapped_idx = final_mapped[orig_idx]
        if mapped_idx == -1:
            print(f"  Skipping '{COLORS_PALETTE[orig_idx][3]}'")
            continue

        mask = (
            (closest_indices_img == orig_idx) & (fg_alpha > 0)
        ).astype(np.uint8) * 255

        # Validate contours exist
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not any(cv2.contourArea(c) >= MIN_CONTOUR_AREA for c in contours):
            continue

        layers_data.append({
            "orig_idx": orig_idx,
            "mapped_idx": mapped_idx,
            "mode": final_modes.get(orig_idx, "fill"),
            "mask": mask,
        })

    if not layers_data:
        print("No drawable layers remain after filtering. Nothing to draw.")
        return

    # ------------------------------------------------------------------
    # 9.5 Optional eraser pass — clean up unwanted pixels before drawing
    # ------------------------------------------------------------------
    print("\n🖌  Opening Eraser Editor (press ENTER to skip, ESC to cancel drawing)...")
    erased = run_eraser_editor(
        layers_data,
        closest_indices_img,
        fg_alpha,
        final_mapped,
        final_modes,
    )
    if erased is None:
        # ESC in eraser = cancel everything
        print("Drawing cancelled from eraser.")
        return
    layers_data = erased

    # ------------------------------------------------------------------
    # 10. Countdown and draw
    # ------------------------------------------------------------------
    countdown = int(config["drawing"].get("pre_draw_countdown", 3))
    print(f"\n\U0001f3a8 Make sure Instagram Draw is open and scroll to Page 1!")
    print(f"   Starting in {countdown} seconds...")
    time.sleep(countdown)

    execute_drawing(
        adb=adb,
        config=config,
        layers_data=layers_data,
        x_phone=x_phone,
        y_phone=y_phone,
        w_phone=w_phone,
        h_phone=h_phone,
        bg_color_idx=bg_color_idx,
    )

    print("\n✅ Drawing completed successfully!")


if __name__ == "__main__":
    main()
