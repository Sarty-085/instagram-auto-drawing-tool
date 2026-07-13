"""drawing_engine — Core drawing logic for the Instagram auto-drawing bot.

Provides the complete pipeline for turning quantised colour layers into ADB
swipe/tap sequences on the phone's Instagram drawing canvas:

* Dynamic swipe-duration calculation based on pixel distance.
* Safety transforms that prevent accidental slider activation.
* Scanline-based shape filling (zig-zag horizontal sweeps).
* Contour-tracing with adjustable polygon approximation.
* Palette page navigation, colour selection, and brush-size selection.
* A top-level ``execute_drawing`` loop that orchestrates everything.

All touch interactions are routed through :class:`adb_utils.ADBConnection`
and all tuneable parameters are read from a *config* dictionary whose
schema matches :data:`config.DEFAULT_CONFIG`.
"""

from __future__ import annotations

import math
import time
from typing import List, Tuple

import cv2
import numpy as np

from adb_utils import ADBConnection
from color_mapping import COLORS_PALETTE


# -----------------------------------------------------------------------
# Swipe duration
# -----------------------------------------------------------------------

def get_swipe_duration(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    config: dict,
) -> int:
    """Calculate an ADB swipe duration proportional to the pixel distance.

    Three speed tiers are defined by the ``drawing`` section of *config*:

    ========== ===================== ==============================
    Distance   Config key            Typical default (ms)
    ========== ===================== ==============================
    > 150 px   swipe_duration_long   800
    > 50 px    swipe_duration_mid    300
    ≤ 50 px    swipe_duration_short  120
    ========== ===================== ==============================

    Parameters
    ----------
    x1, y1 : int
        Start coordinates.
    x2, y2 : int
        End coordinates.
    config : dict
        Full application config (must contain ``config['drawing']``).

    Returns
    -------
    int
        Duration in milliseconds.
    """
    dist: float = math.hypot(x2 - x1, y2 - y1)
    drawing_cfg: dict = config["drawing"]

    if dist > 150:
        return int(drawing_cfg["swipe_duration_long"])
    if dist > 50:
        return int(drawing_cfg["swipe_duration_mid"])
    return int(drawing_cfg["swipe_duration_short"])


# -----------------------------------------------------------------------
# Coordinate safety helpers
# -----------------------------------------------------------------------

def make_swipe_coordinates_safe(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    safe_x: int = 120,
) -> Tuple[int, int, int, int]:
    """Prevent the touch-down point from landing inside the slider zone.

    Instagram's brush-size slider occupies the left edge of the screen
    (``X < safe_x``).  Starting a swipe there would accidentally resize
    the brush instead of drawing.

    Rules
    -----
    1. If *x1* is in the danger zone but *x2* is safe — reverse the
       swipe direction (the visible line is the same).
    2. If **both** points are in the danger zone — shift *x1* to
       ``safe_x`` so the touch-down is outside the slider.
    3. Otherwise return the coordinates unchanged.

    Parameters
    ----------
    x1, y1, x2, y2 : int
        Original swipe coordinates.
    safe_x : int, optional
        The X boundary of the slider zone (default ``120``).

    Returns
    -------
    tuple[int, int, int, int]
        The (possibly adjusted) ``(x1, y1, x2, y2)`` values.
    """
    if x1 < safe_x and x2 >= safe_x:
        # Case 1: reverse direction so touch-down is on the safe side.
        return x2, y2, x1, y1
    if x1 < safe_x and x2 < safe_x:
        # Case 2: both unsafe — nudge x1 out of the zone.
        return safe_x, y1, x2, y2
    # Case 3: already safe.
    return x1, y1, x2, y2


def make_contour_safe(
    path: np.ndarray,
    x_phone: int,
    safe_x: int = 120,
) -> np.ndarray:
    """Roll a closed contour so its first point is outside the slider zone.

    ``path`` is an (N, 1, 2) or (N, 2) array of points expressed in
    *mask-local* coordinates.  The actual phone X of a point is
    ``point_x + x_phone``.  This function finds the first point whose
    phone-X is ≥ ``safe_x`` and :func:`numpy.roll` s the array so that
    point becomes the new index 0.

    If no safe point exists the array is returned as-is.

    Parameters
    ----------
    path : np.ndarray
        Contour points, shape ``(N, 1, 2)`` or ``(N, 2)``.
    x_phone : int
        Horizontal offset from mask coordinates to phone coordinates.
    safe_x : int, optional
        Slider-zone boundary (default ``120``).

    Returns
    -------
    np.ndarray
        The rolled contour (same shape as input).
    """
    squeezed = path.reshape(-1, 2)

    for idx, point in enumerate(squeezed):
        if point[0] + x_phone >= safe_x:
            rolled = np.roll(squeezed, -idx, axis=0)
            return rolled.reshape(path.shape)

    # Every point is inside the slider zone — nothing we can do.
    return path


# -----------------------------------------------------------------------
# Interior fill via horizontal scanlines
# -----------------------------------------------------------------------

def get_fill_paths_from_mask(
    mask: np.ndarray,
    step_size: int,
) -> List[np.ndarray]:
    """Generate zig-zag horizontal scanlines directly from a binary mask.

    Unlike the old contour-based approach, this function scans the **actual
    mask pixels** row by row and emits one swipe segment per connected
    horizontal run of non-zero pixels.  This correctly handles masks with
    interior holes (e.g. a ring-shaped black outline whose interior pixels
    belong to a different colour layer): those interior pixels are zero in
    the mask, so no swipe is emitted for them — the fill cannot bleed into
    areas that were not originally that colour.

    Even-indexed segments run left → right; odd-indexed ones run right →
    left, producing a zig-zag pattern that minimises pen-lift travel.

    Parameters
    ----------
    mask : np.ndarray
        Binary single-channel mask (uint8, shape H×W), values 0 or 255.
    step_size : int
        Vertical distance (pixels) between consecutive scanline rows.

    Returns
    -------
    list[np.ndarray]
        Each element is an int32 array of shape ``(2, 1, 2)`` representing
        a start and end point in canvas-local coordinates:
        ``[[[x_start, y]], [[x_end, y]]]``.
    """
    h_mask, w_mask = mask.shape[:2]
    paths: List[np.ndarray] = []
    line_index = 0

    for row in range(0, h_mask, step_size):
        filled = np.where(mask[row] > 0)[0]
        if filled.size < 2:
            continue  # 0 or 1 pixel → nothing drawable

        # Find connected runs: consecutive columns with no gap > 1
        breaks = np.where(np.diff(filled) > 1)[0]
        run_start_idxs = np.concatenate([[0], breaks + 1])
        run_end_idxs   = np.concatenate([breaks, [filled.size - 1]])

        for rs_i, re_i in zip(run_start_idxs, run_end_idxs):
            x_left  = int(filled[rs_i])
            x_right = int(filled[re_i])

            if x_right <= x_left:
                continue  # single-pixel-wide run — skip

            if line_index % 2 == 0:
                start, end = (x_left, row), (x_right, row)
            else:
                start, end = (x_right, row), (x_left, row)

            paths.append(
                np.array([[[start[0], start[1]]], [[end[0], end[1]]]], dtype=np.int32)
            )
            line_index += 1

    return paths


def get_fill_paths_for_contour(
    contour: np.ndarray,
    step_size: int,
    w_mask: int,
    h_mask: int,
) -> List[np.ndarray]:
    """.. deprecated::
        Use :func:`get_fill_paths_from_mask` instead.

    Kept for backward compatibility.  Generates zig-zag scanlines by
    filling the contour's bounding box — this can bleed into interior holes
    in ring-shaped contours, which is why it is deprecated.
    """
    bx, by, bw, bh = cv2.boundingRect(contour)
    bx = max(0, bx); by = max(0, by)
    bw = min(bw, w_mask - bx); bh = min(bh, h_mask - by)
    if bw <= 0 or bh <= 0:
        return []
    local_contour = contour - np.array([[[bx, by]]], dtype=np.int32)
    local_mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.drawContours(local_mask, [local_contour], -1, color=255, thickness=cv2.FILLED)
    return get_fill_paths_from_mask(local_mask, step_size)




# -----------------------------------------------------------------------
# Palette and brush selection
# -----------------------------------------------------------------------

def select_color(
    adb: ADBConnection,
    palette_idx: int,
    current_page: List[int],
    config: dict,
) -> None:
    """Navigate to the correct palette page and tap a colour swatch.

    ``current_page`` is a **mutable** one-element list (e.g. ``[1]``)
    so the caller's state is updated in-place when pages change.

    The function swipes left or right using the coordinates in
    ``config['palette_page_swipe']`` until the target page is reached,
    then taps the swatch's X coordinate at
    ``config['device']['palette_y']``.

    Parameters
    ----------
    adb : ADBConnection
        Active ADB session.
    palette_idx : int
        Index into :data:`color_mapping.COLORS_PALETTE`.
    current_page : list[int]
        Single-element list holding the currently visible page number.
    config : dict
        Full application config.
    """
    _bgr, target_page, color_x, color_name = COLORS_PALETTE[palette_idx]

    swipe_cfg = config["palette_page_swipe"]
    palette_y: int = int(config["device"]["palette_y"])
    start_x: int = int(swipe_cfg["start_x"])
    end_x: int = int(swipe_cfg["end_x"])
    duration: int = int(swipe_cfg["duration"])
    settle: float = float(swipe_cfg["settle_delay"])

    # If it is a custom spectrum colour
    if target_page == -1:
        # 1. Swipe back to Page 1 (spectrum picker starts there)
        while current_page[0] != 1:
            if current_page[0] < 1:
                adb.swipe(start_x, palette_y, end_x, palette_y, duration)
                current_page[0] += 1
            else:
                adb.swipe(end_x, palette_y, start_x, palette_y, duration)
                current_page[0] -= 1
            time.sleep(settle)

        # 2. Long-press the first swatch on Page 1 and drag to the target X on the spectrum
        x_positions = config["device"].get("palette_x_positions", [432, 515, 597, 680, 763, 845, 928])
        swatch_x = int(x_positions[0])
        spectrum_y = int(config["spectrum"]["y"])
        target_x = int(color_x)

        # Slow swipe/drag gesture (1.5 seconds)
        adb.swipe(swatch_x, palette_y, target_x, spectrum_y, 1500)
        # Settle to register the touch up
        time.sleep(0.5)
        return

    # Swipe to the correct page for standard colours.
    while current_page[0] != target_page:
        if current_page[0] < target_page:
            # Swipe left → next page (start_x > end_x).
            adb.swipe(start_x, palette_y, end_x, palette_y, duration)
            current_page[0] += 1
        else:
            # Swipe right → previous page (end_x → start_x).
            adb.swipe(end_x, palette_y, start_x, palette_y, duration)
            current_page[0] -= 1
        time.sleep(settle)

    # Use calibrated horizontal coordinates
    x_positions = config["device"].get("palette_x_positions", [432, 515, 597, 680, 763, 845, 928])
    color_x = x_positions[palette_idx % 7]

    # Tap the colour swatch.
    adb.tap(color_x, palette_y)


def select_brush_size(
    adb: ADBConnection,
    size_number: int,
    config: dict,
) -> None:
    """Drag the brush-size slider to the position for *size_number*.

    Reads the target Y from ``config['brush_config'][str(size_number)]['y']``
    and performs a short horizontal swipe on the slider rail at
    ``config['device']['brush_slider_x']`` to activate the position.

    Parameters
    ----------
    adb : ADBConnection
        Active ADB session.
    size_number : int
        Brush preset (``1`` – ``5``).
    config : dict
        Full application config.
    """
    brush_x: int = int(config["device"]["brush_slider_x"])
    target_y: int = int(config["brush_config"][str(size_number)]["y"])

    # A tiny horizontal drag at the target Y to "click" the slider.
    adb.swipe(brush_x, target_y, brush_x + 15, target_y, 200)


# -----------------------------------------------------------------------
# Main drawing loop
# -----------------------------------------------------------------------

def execute_drawing(
    adb: ADBConnection,
    config: dict,
    layers_data: List[dict],
    x_phone: int,
    y_phone: int,
    w_phone: int,
    h_phone: int,
    bg_color_idx: int,
) -> None:
    """Execute the full drawing sequence on the phone canvas.

    Parameters
    ----------
    adb : ADBConnection
        Active ADB session.
    config : dict
        Full application config.
    layers_data : list[dict]
        Ordered drawing layers.  Each dict has keys:

        * ``orig_idx`` – original palette index of this colour.
        * ``mapped_idx`` – palette index after any remapping.
        * ``mode`` – ``'fill'`` or ``'outline'``.
        * ``mask`` – ``np.ndarray`` (uint8, single-channel) binary mask.
    x_phone, y_phone : int
        Top-left corner of the drawing area on the phone screen.
    w_phone, h_phone : int
        Dimensions of the drawing area on the phone screen.
    bg_color_idx : int
        Palette index for the background colour, or ``-1`` to skip
        background filling.
    """
    drawing_cfg: dict = config["drawing"]
    safe_x: int = int(config["device"].get("safe_x_boundary", 120))
    inter_delay: float = float(drawing_cfg.get("inter_swipe_delay", 0.02))

    current_page: List[int] = [1]

    # ------------------------------------------------------------------
    # a) Background fill
    # ------------------------------------------------------------------
    if bg_color_idx != -1:
        _bgr, _page, _cx, bg_name = COLORS_PALETTE[bg_color_idx]
        print(f"[Background] Filling with {bg_name}")

        select_color(adb, bg_color_idx, current_page, config)
        select_brush_size(adb, 5, config)

        bg_step = 18
        line_index = 0
        for row_offset in range(0, h_phone, bg_step):
            y = y_phone + row_offset
            if line_index % 2 == 0:
                sx, sy, ex, ey = x_phone, y, x_phone + w_phone, y
            else:
                sx, sy, ex, ey = x_phone + w_phone, y, x_phone, y

            sx, sy, ex, ey = make_swipe_coordinates_safe(
                sx, sy, ex, ey, safe_x,
            )
            dur = get_swipe_duration(sx, sy, ex, ey, config)
            adb.swipe(sx, sy, ex, ey, dur)
            time.sleep(inter_delay)
            line_index += 1

        settle = float(drawing_cfg.get("post_background_settle", 1.5))
        time.sleep(settle)

    # ------------------------------------------------------------------
    # b) Layer-by-layer drawing
    # ------------------------------------------------------------------
    total_layers = len(layers_data)

    for layer_num, layer in enumerate(layers_data, start=1):
        palette_idx: int = layer["mapped_idx"]
        mode: str = layer["mode"]
        mask: np.ndarray = layer["mask"]

        _bgr, _page, _cx, color_name = COLORS_PALETTE[palette_idx]
        print(
            f"[Layer {layer_num}/{total_layers}] "
            f"Drawing {color_name} - {mode.upper()} mode"
        )

        select_color(adb, palette_idx, current_page, config)
        select_brush_size(adb, 1, config)

        h_mask, w_mask = mask.shape[:2]

        if mode == "fill":
            # --- Scanline fill (mask-direct: respects holes correctly) ------
            fill_step: int = int(drawing_cfg["fill_step_size"])
            segments = get_fill_paths_from_mask(mask, fill_step)
            for seg in segments:
                pt_start = seg[0][0]
                pt_end   = seg[1][0]
                sx = int(pt_start[0]) + x_phone
                sy = int(pt_start[1]) + y_phone
                ex = int(pt_end[0])   + x_phone
                ey = int(pt_end[1])   + y_phone

                sx, sy, ex, ey = make_swipe_coordinates_safe(sx, sy, ex, ey, safe_x)
                dur = get_swipe_duration(sx, sy, ex, ey, config)
                adb.swipe(sx, sy, ex, ey, dur)
                time.sleep(inter_delay)
            # Settle after the full fill so the pen lift is fully registered
            time.sleep(max(0.15, inter_delay * 2))


        else:
            # --- Outline tracing -----------------------------------------
            epsilon_factor: float = float(drawing_cfg["contour_epsilon"])
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            for cnt in contours:
                perimeter = cv2.arcLength(cnt, closed=True)
                approx = cv2.approxPolyDP(
                    cnt, epsilon_factor * perimeter, closed=True,
                )
                approx = make_contour_safe(approx, x_phone, safe_x)
                points = approx.reshape(-1, 2)

                for i in range(len(points)):
                    pt1 = points[i]
                    pt2 = points[(i + 1) % len(points)]

                    sx = int(pt1[0]) + x_phone
                    sy = int(pt1[1]) + y_phone
                    ex = int(pt2[0]) + x_phone
                    ey = int(pt2[1]) + y_phone

                    sx, sy, ex, ey = make_swipe_coordinates_safe(
                        sx, sy, ex, ey, safe_x,
                    )
                    dur = get_swipe_duration(sx, sy, ex, ey, config)
                    adb.swipe(sx, sy, ex, ey, dur)
                    time.sleep(inter_delay)
                # Extra settle time after finishing a contour to guarantee the lift is registered
                time.sleep(max(0.15, inter_delay * 2))

    print("[Done] All layers drawn.")
