"""gui — OpenCV-based GUI for the Instagram auto-drawing bot.

Provides two interactive panels:

1. **Overlay Editor** — lets the user drag and resize the source image
   on top of a phone screenshot to choose the drawing position.
2. **Mapping Dashboard** — a 960×640 panel where the user can reorder
   layers, remap colours, toggle fill/outline modes, and configure
   a background colour.

All mutable state is encapsulated in closures or helper classes
rather than module-level globals.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from color_mapping import COLORS_PALETTE


# -----------------------------------------------------------------------
# Alpha-blending helper
# -----------------------------------------------------------------------

def overlay_image_bgra(
    background: np.ndarray,
    foreground: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> np.ndarray:
    """Alpha-blend a BGRA *foreground* onto a BGR *background*.

    The foreground is resized to ``(w, h)`` and composited at position
    ``(x, y)`` on a copy of the background.

    Parameters
    ----------
    background : np.ndarray
        BGR image (H×W×3).
    foreground : np.ndarray
        BGRA image with alpha channel.
    x, y : int
        Top-left corner on the background.
    w, h : int
        Target width and height for the foreground.

    Returns
    -------
    np.ndarray
        A copy of *background* with the blended foreground.
    """
    fg_resized = cv2.resize(foreground, (w, h))
    bg_h, bg_w = background.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)

    if x1 >= x2 or y1 >= y2:
        return background.copy()

    fg_x1, fg_y1 = x1 - x, y1 - y
    fg_x2, fg_y2 = fg_x1 + (x2 - x1), fg_y1 + (y2 - y1)

    out = background.copy()
    fg_crop = fg_resized[fg_y1:fg_y2, fg_x1:fg_x2]
    bg_crop = out[y1:y2, x1:x2]

    alpha = fg_crop[:, :, 3:4] / 255.0
    color = fg_crop[:, :, :3]

    bg_crop[:] = (color * alpha + bg_crop * (1 - alpha)).astype(np.uint8)
    return out


# -----------------------------------------------------------------------
# Overlay Editor
# -----------------------------------------------------------------------

def run_overlay_editor(
    screenshot: np.ndarray,
    source_bgra: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """Interactive overlay placement window.

    Shows the phone *screenshot* with *source_bgra* overlaid.  The user
    can drag to move and drag the bottom-right red corner to resize
    (aspect-ratio locked).

    Parameters
    ----------
    screenshot : np.ndarray
        Phone screenshot, BGR.
    source_bgra : np.ndarray
        Source drawing image, BGRA.

    Returns
    -------
    tuple[int, int, int, int] or None
        ``(x, y, w, h)`` in **display space** when confirmed, or
        ``None`` if the user pressed ESC.
    """
    h_screen, w_screen = screenshot.shape[:2]
    src_h, src_w = source_bgra.shape[:2]

    # Scale screenshot to fit laptop screens
    max_display_h = 800
    scale = min(1.0, max_display_h / h_screen)
    display_w = int(w_screen * scale)
    display_h = int(h_screen * scale)
    display_screenshot = cv2.resize(screenshot, (display_w, display_h))

    # Initial overlay position and size
    ov_w = int(display_w * 0.5)
    ov_h = int(ov_w * (src_h / src_w))
    ov_x = int((display_w - ov_w) / 2)
    ov_y = int((display_h - ov_h) / 2)

    # Mutable state in a dict (closure-friendly)
    state = {
        "dragging": False,
        "resizing": False,
        "start_mx": 0,
        "start_my": 0,
        "start_ox": 0,
        "start_oy": 0,
        "start_ow": 0,
        "start_oh": 0,
        "ov_x": ov_x,
        "ov_y": ov_y,
        "ov_w": ov_w,
        "ov_h": ov_h,
    }

    handle_size = 15

    def _on_mouse(event: int, mx: int, my: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy, ow, oh = state["ov_x"], state["ov_y"], state["ov_w"], state["ov_h"]
            if (ox + ow - handle_size <= mx <= ox + ow + handle_size and
                    oy + oh - handle_size <= my <= oy + oh + handle_size):
                state["resizing"] = True
                state["start_mx"], state["start_my"] = mx, my
                state["start_ow"], state["start_oh"] = ow, oh
            elif ox <= mx <= ox + ow and oy <= my <= oy + oh:
                state["dragging"] = True
                state["start_mx"], state["start_my"] = mx, my
                state["start_ox"], state["start_oy"] = ox, oy

        elif event == cv2.EVENT_MOUSEMOVE:
            if state["dragging"]:
                dx = mx - state["start_mx"]
                dy = my - state["start_my"]
                state["ov_x"] = state["start_ox"] + dx
                state["ov_y"] = state["start_oy"] + dy
            elif state["resizing"]:
                dx = mx - state["start_mx"]
                new_w = max(50, state["start_ow"] + dx)
                aspect = state["start_oh"] / state["start_ow"]
                state["ov_w"] = new_w
                state["ov_h"] = int(new_w * aspect)

        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
            state["resizing"] = False

    window_name = "Overlay Editor (ENTER to Confirm, ESC to Cancel)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _on_mouse)

    print("\n--- OVERLAY EDITOR ---")
    print("  Drag to Move | Drag Red corner to Resize")
    print("  ENTER/SPACE to Confirm | ESC to Cancel")

    while True:
        canvas = display_screenshot.copy()
        ox, oy, ow, oh = state["ov_x"], state["ov_y"], state["ov_w"], state["ov_h"]

        canvas = overlay_image_bgra(canvas, source_bgra, ox, oy, ow, oh)

        # Green bounding border
        cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2)
        # Red resize handle
        cv2.rectangle(canvas, (ox + ow - 6, oy + oh - 6),
                      (ox + ow + 6, oy + oh + 6), (0, 0, 255), -1)

        # Guide text with shadow
        for text, ty in [
            ("Drag body to Move | Drag Red corner to Resize", 30),
            ("Press ENTER/SPACE to Confirm | Press ESC to Cancel", 55),
        ]:
            cv2.putText(canvas, text, (15, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (15, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(15) & 0xFF
        if key in (13, 32):  # Enter or Space
            cv2.destroyAllWindows()
            return (ox, oy, ow, oh, scale)
        elif key in (27, ord('q'), ord('c')):  # ESC, q, c
            print("Placement cancelled by user.")
            cv2.destroyAllWindows()
            return None



# -----------------------------------------------------------------------
# Eraser editor
# -----------------------------------------------------------------------

def run_eraser_editor(
    layers_data: List[dict],
    closest_indices_img: np.ndarray,
    fg_alpha: np.ndarray,
    mapped_indices: List[int],
    layer_modes: dict,
) -> Optional[List[dict]]:
    """Interactive per-layer eraser panel.

    Displays a composite colour preview of the image and lets the user
    paint over areas they **do not** want drawn.  Left-click erases (zeros
    out mask pixels); right-click restores them from the original mask.

    The eraser brush size is adjustable with the ``[`` and ``]`` keys.
    Press ``N`` / ``P`` to cycle through layers.  Press ENTER to confirm
    all edits and return the modified layer list.  Press ESC to cancel
    (original masks are left untouched).

    Parameters
    ----------
    layers_data : list[dict]
        The list of layer dicts produced by ``draw_interactive.main``.
        Each dict must have ``orig_idx``, ``mapped_idx``, ``mode``,
        and ``mask`` keys.  Masks are **copied** before editing so the
        caller's originals are never mutated on cancel.
    closest_indices_img : np.ndarray
        Per-pixel palette index image (H×W int).
    fg_alpha : np.ndarray
        Foreground alpha mask (H×W uint8).
    mapped_indices : list[int]
        Current colour remapping (index i → palette index).
    layer_modes : dict
        ``{palette_idx: 'fill' | 'outline'}`` per layer.

    Returns
    -------
    list[dict] or None
        The modified layer list on confirm, or ``None`` if cancelled.
    """
    if not layers_data:
        return layers_data

    # Work on deep copies of masks so we can discard on cancel.
    import copy as _copy
    working_layers = _copy.deepcopy(layers_data)

    # --- Display scale ---------------------------------------------------
    h_mask, w_mask = working_layers[0]["mask"].shape[:2]
    MAX_DISP = 720
    scale = min(1.0, MAX_DISP / max(h_mask, w_mask))
    dw = max(1, int(w_mask * scale))
    dh = max(1, int(h_mask * scale))

    state = {
        "layer_idx": 0,        # which layer is currently active
        "brush_r": 20,         # eraser radius in display pixels
        "painting": False,
        "restoring": False,
        "mx": 0,
        "my": 0,
    }

    def _composite_preview(active_li: int) -> np.ndarray:
        """Build a BGR composite; active layer at full brightness, others dimmed."""
        canvas = np.zeros((h_mask, w_mask, 3), dtype=np.uint8)
        for li, lyr in enumerate(working_layers):
            m_idx = lyr["mapped_idx"]
            color = np.array(COLORS_PALETTE[m_idx][0], dtype=np.float32)
            mask_bool = lyr["mask"] > 0
            if li != active_li:
                color = color * 0.25
            canvas[mask_bool] = color.astype(np.uint8)
        return canvas

    def _apply_brush(lyr: dict, orig_mask: np.ndarray, mx_d: int, my_d: int, erase: bool) -> None:
        """Paint or restore a circular region on the working mask (vectorised)."""
        # Convert display coords → mask coords
        cx = int(mx_d / scale)
        cy = int(my_d / scale)
        r = max(1, int(state["brush_r"] / scale))

        # Bounding box clamped to mask dimensions
        y0 = max(0, cy - r)
        y1 = min(h_mask, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(w_mask, cx + r + 1)

        if y0 >= y1 or x0 >= x1:
            return

        # Build circular boolean mask using numpy (fast, no Python loops)
        ys = np.arange(y0, y1) - cy
        xs = np.arange(x0, x1) - cx
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        circle = (grid_y ** 2 + grid_x ** 2) <= r * r

        region = lyr["mask"][y0:y1, x0:x1]
        if erase:
            region[circle] = 0
        else:
            region[circle] = orig_mask[y0:y1, x0:x1][circle]

    # Keep original masks for restore (right-click)
    orig_masks = [lyr["mask"].copy() for lyr in working_layers]

    def _on_mouse(event: int, mx: int, my: int, flags: int, param: object) -> None:
        li = state["layer_idx"]
        lyr = working_layers[li]
        orig = orig_masks[li]

        if event == cv2.EVENT_LBUTTONDOWN:
            state["painting"] = True
            _apply_brush(lyr, orig, mx, my, erase=True)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["restoring"] = True
            _apply_brush(lyr, orig, mx, my, erase=False)
        elif event == cv2.EVENT_MOUSEMOVE:
            state["mx"], state["my"] = mx, my
            if state["painting"]:
                _apply_brush(lyr, orig, mx, my, erase=True)
            elif state["restoring"]:
                _apply_brush(lyr, orig, mx, my, erase=False)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            state["painting"] = False
            state["restoring"] = False

    window = "Eraser Editor — L:erase  R:restore  [/]:brush size  N/P:layer  ENTER:done  ESC:cancel"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)

    total = len(working_layers)
    print("\n--- ERASER EDITOR ---")
    print("  Left-click  : erase pixels from current layer")
    print("  Right-click : restore erased pixels")
    print("  [ / ]       : shrink / grow eraser brush")
    print("  N / P       : next / previous layer")
    print("  ENTER/SPACE : confirm all edits")
    print("  ESC         : cancel (no changes applied)")

    while True:
        li = state["layer_idx"]
        lyr = working_layers[li]
        m_idx = lyr["mapped_idx"]
        color_name = COLORS_PALETTE[m_idx][3]

        # Build display
        preview = _composite_preview(li)
        disp = cv2.resize(preview, (dw, dh), interpolation=cv2.INTER_NEAREST)

        # Draw cursor circle
        mx_d, my_d = state["mx"], state["my"]
        cv2.circle(disp, (mx_d, my_d), state["brush_r"], (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(disp, (mx_d, my_d), state["brush_r"], (0, 0, 0), 2, cv2.LINE_AA)

        # HUD — layer info bar at the top
        hud_h = 60
        hud = np.zeros((hud_h, dw, 3), dtype=np.uint8)
        hud[:] = (30, 30, 30)
        swatch_color = tuple(int(c) for c in COLORS_PALETTE[m_idx][0])
        cv2.rectangle(hud, (8, 8), (44, 52), swatch_color, -1)
        cv2.rectangle(hud, (8, 8), (44, 52), (200, 200, 200), 1)
        layer_text = f"Layer {li + 1}/{total}: {color_name}  |  Brush: {state['brush_r']}px"
        cv2.putText(hud, layer_text, (54, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        hint = "L=erase  R=restore  [/]=brush  N/P=layer  ENTER=done  ESC=cancel"
        cv2.putText(hud, hint, (54, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

        combined = np.vstack([hud, disp])
        cv2.imshow(window, combined)

        key = cv2.waitKey(15) & 0xFF
        if key in (13, 32):  # ENTER / SPACE — confirm
            cv2.destroyAllWindows()
            print("  Eraser edits confirmed.")
            return working_layers
        elif key == 27:  # ESC — cancel
            cv2.destroyAllWindows()
            print("  Eraser cancelled — original masks kept.")
            return None
        elif key == ord('n') or key == ord('N'):
            state["layer_idx"] = (li + 1) % total
            print(f"  → Layer {state['layer_idx'] + 1}/{total}")
        elif key == ord('p') or key == ord('P'):
            state["layer_idx"] = (li - 1) % total
            print(f"  → Layer {state['layer_idx'] + 1}/{total}")
        elif key == ord(']'):
            state["brush_r"] = min(150, state["brush_r"] + 5)
        elif key == ord('['):
            state["brush_r"] = max(3, state["brush_r"] - 5)


# -----------------------------------------------------------------------
# Preview rendering
# -----------------------------------------------------------------------

def get_preview_image(
    closest_indices_img: np.ndarray,
    fg_alpha: np.ndarray,
    present_indices: List[int],
    mapped_indices: List[int],
    layer_modes: dict,
    selected_idx: Union[int, str, None],
    bg_color_idx: int,
) -> np.ndarray:
    """Composite a live preview image highlighting the selected layer.

    The background is drawn first (if enabled), then foreground layers
    are composited in ``present_indices`` order.  The selected layer
    is at full brightness; all others are dimmed to 20%.

    Returns
    -------
    np.ndarray
        BGR preview image, same shape as *closest_indices_img*.
    """
    h, w = closest_indices_img.shape
    preview = np.zeros((h, w, 3), dtype=np.uint8)

    # Background rectangle
    if bg_color_idx != -1:
        bg_bgr = np.array(COLORS_PALETTE[bg_color_idx][0], dtype=np.float32)
        if selected_idx != "bg":
            bg_bgr = bg_bgr * 0.2
        preview[:, :] = bg_bgr.astype(np.uint8)

    selected_orig = (
        present_indices[selected_idx]
        if isinstance(selected_idx, int) and selected_idx < len(present_indices)
        else None
    )

    # Foreground layers
    for orig_idx in present_indices:
        mapped_idx = mapped_indices[orig_idx]
        if mapped_idx == -1:
            continue

        layer_mask = (closest_indices_img == orig_idx) & (fg_alpha > 0)

        if layer_modes.get(orig_idx) == "outline":
            m8 = layer_mask.astype(np.uint8)
            kern = np.ones((3, 3), np.uint8)
            draw_mask = (cv2.dilate(m8, kern) - cv2.erode(m8, kern)) > 0
        else:
            draw_mask = layer_mask

        color_bgr = np.array(COLORS_PALETTE[mapped_idx][0], dtype=np.float32)

        # Dim non-selected layers
        if selected_idx == "bg":
            color_bgr *= 0.2
        elif selected_orig is not None and orig_idx != selected_orig:
            color_bgr *= 0.2

        preview[draw_mask] = color_bgr.astype(np.uint8)

    return preview


# -----------------------------------------------------------------------
# Dashboard internal renderers
# -----------------------------------------------------------------------

def _draw_row(
    canvas: np.ndarray,
    idx: int,
    orig_idx: int,
    y_min: int,
    y_max: int,
    is_selected: bool,
    mapped_indices: List[int],
    layer_modes: dict,
) -> None:
    """Render a single layer row with mode toggle and reorder buttons."""
    bg = (75, 75, 75) if is_selected else (45, 43, 42)
    border = (0, 255, 255) if is_selected else (80, 80, 80)

    cv2.rectangle(canvas, (485, y_min), (725, y_max), bg, -1)
    cv2.rectangle(canvas, (485, y_min), (725, y_max), border, 2 if is_selected else 1)

    # Original swatch
    cv2.rectangle(canvas, (492, y_min + 7), (512, y_max - 7), COLORS_PALETTE[orig_idx][0], -1)
    cv2.rectangle(canvas, (492, y_min + 7), (512, y_max - 7), (255, 255, 255), 1)

    # Arrow
    cv2.putText(canvas, ">", (518, y_min + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)

    # Mapped swatch or SKIP
    m_idx = mapped_indices[orig_idx]
    if m_idx == -1:
        cv2.putText(canvas, "SKIP", (530, y_min + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (530, y_min + 7), (550, y_max - 7), COLORS_PALETTE[m_idx][0], -1)
        cv2.rectangle(canvas, (530, y_min + 7), (550, y_max - 7), (255, 255, 255), 1)

    # Mode pill
    mode = layer_modes.get(orig_idx, "fill")
    mode_text = "FILL" if mode == "fill" else "LINE"
    mode_bg = (0, 90, 0) if mode == "fill" else (100, 40, 0)
    cv2.rectangle(canvas, (565, y_min + 6), (635, y_max - 6), mode_bg, -1)
    cv2.rectangle(canvas, (565, y_min + 6), (635, y_max - 6), (200, 200, 200), 1)
    cv2.putText(canvas, mode_text, (585, y_min + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # Up button
    cv2.rectangle(canvas, (655, y_min + 6), (685, y_max - 6), (60, 60, 60), -1)
    cv2.rectangle(canvas, (655, y_min + 6), (685, y_max - 6), (200, 200, 200), 1)
    cv2.putText(canvas, "^", (667, y_min + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # Down button
    cv2.rectangle(canvas, (695, y_min + 6), (725, y_max - 6), (60, 60, 60), -1)
    cv2.rectangle(canvas, (695, y_min + 6), (725, y_max - 6), (200, 200, 200), 1)
    cv2.putText(canvas, "v", (707, y_min + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_dashboard_canvas(
    preview_img: np.ndarray,
    present_indices: List[int],
    selected_idx: Union[int, str, None],
    mapped_indices: List[int],
    layer_modes: dict,
    scroll_off: int,
    bg_color_idx: int,
) -> np.ndarray:
    """Render the full 960×640 dashboard canvas."""
    canvas = np.zeros((640, 960, 3), dtype=np.uint8)
    canvas[:] = (34, 32, 31)

    # Title
    cv2.putText(canvas, "INSTAGRAM DOODLE BOT - DASHBOARD", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Grid lines
    cv2.line(canvas, (475, 70), (475, 570), (70, 70, 70), 1)
    cv2.line(canvas, (735, 70), (735, 570), (70, 70, 70), 1)
    cv2.line(canvas, (20, 570), (940, 570), (70, 70, 70), 1)

    # Sub-headings
    cv2.putText(canvas, "Live Preview", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Layers Config", (485, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Instagram Palette", (745, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # Preview
    preview_resized = cv2.resize(preview_img, (440, 440))
    canvas[85:525, 20:460] = preview_resized
    cv2.rectangle(canvas, (20, 85), (460, 525), (100, 100, 100), 2)

    # Background row
    is_bg_sel = selected_idx == "bg"
    bg_row_bg = (85, 85, 85) if is_bg_sel else (45, 43, 42)
    bg_border = (0, 255, 255) if is_bg_sel else (80, 80, 80)

    cv2.rectangle(canvas, (485, 80), (725, 118), bg_row_bg, -1)
    cv2.rectangle(canvas, (485, 80), (725, 118), bg_border, 2 if is_bg_sel else 1)

    if bg_color_idx == -1:
        cv2.rectangle(canvas, (492, 87), (512, 111), (30, 30, 30), -1)
        cv2.rectangle(canvas, (492, 87), (512, 111), (100, 100, 100), 1)
        cv2.line(canvas, (492, 87), (512, 111), (0, 0, 255), 2)
    else:
        cv2.rectangle(canvas, (492, 87), (512, 111), COLORS_PALETTE[bg_color_idx][0], -1)
        cv2.rectangle(canvas, (492, 87), (512, 111), (255, 255, 255), 1)

    cv2.putText(canvas, ">", (518, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(canvas, "BACKGROUND", (530, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # BG mode pill (always FILL)
    cv2.rectangle(canvas, (565, 86), (635, 112), (0, 90, 0), -1)
    cv2.rectangle(canvas, (565, 86), (635, 112), (200, 200, 200), 1)
    cv2.putText(canvas, "FILL", (585, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.line(canvas, (485, 122), (725, 122), (90, 90, 90), 1)

    # Layer rows
    N = len(present_indices)
    if N <= 9:
        for i in range(N):
            y_min = 125 + i * 46
            y_max = y_min + 38
            _draw_row(canvas, i, present_indices[i], y_min, y_max,
                      selected_idx == i, mapped_indices, layer_modes)
    else:
        # Scroll up button
        up_disabled = scroll_off == 0
        up_bg = (50, 50, 50) if up_disabled else (70, 70, 220)
        cv2.rectangle(canvas, (485, 125), (725, 150), up_bg, -1)
        cv2.rectangle(canvas, (485, 125), (725, 150), (120, 120, 120), 1)
        cv2.putText(canvas, "^ Scroll Up", (565, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(8):
            idx = i + scroll_off
            if idx >= N:
                break
            y_min = 160 + i * 46
            y_max = y_min + 38
            _draw_row(canvas, idx, present_indices[idx], y_min, y_max,
                      selected_idx == idx, mapped_indices, layer_modes)

        # Scroll down button
        dn_disabled = scroll_off >= N - 8
        dn_bg = (50, 50, 50) if dn_disabled else (70, 70, 220)
        cv2.rectangle(canvas, (485, 535), (725, 560), dn_bg, -1)
        cv2.rectangle(canvas, (485, 535), (725, 560), (120, 120, 120), 1)
        cv2.putText(canvas, "v Scroll Down", (560, 552), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Toggle Skip button
    cv2.rectangle(canvas, (485, 580), (725, 615), (40, 40, 180), -1)
    cv2.rectangle(canvas, (485, 580), (725, 615), (255, 255, 255), 1)
    cv2.putText(canvas, "TOGGLE SKIP / INCLUDE", (515, 602), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Palette grid
    for idx, item in enumerate(COLORS_PALETTE[:22]):
        col = idx % 4
        row = idx // 4
        cx = 745 + col * 45 + 22
        cy = 85 + row * 50 + 22

        cv2.circle(canvas, (cx, cy), 18, item[0], -1)
        cv2.circle(canvas, (cx, cy), 18, (255, 255, 255), 1)

        # Selection indicator
        target_matched = False
        if selected_idx == "bg":
            target_matched = bg_color_idx == idx
        elif isinstance(selected_idx, int) and selected_idx < len(present_indices):
            target_matched = mapped_indices[present_indices[selected_idx]] == idx

        if target_matched:
            cv2.circle(canvas, (cx, cy), 6, (0, 255, 0), -1)
            cv2.circle(canvas, (cx, cy), 9, (0, 255, 0), 1)
            cv2.circle(canvas, (cx, cy), 20, (0, 255, 0), 2)

    # Footer
    cv2.putText(canvas, "ENTER: Start drawing on phone | ESC: Cancel drawing", (20, 595),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Click rows to edit. Reorder so background fills draw first.", (20, 615),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    return canvas


# -----------------------------------------------------------------------
# Mapping Dashboard
# -----------------------------------------------------------------------

def run_mapping_dashboard(
    closest_indices_img: np.ndarray,
    fg_alpha: np.ndarray,
    present_indices: List[int],
    layer_modes: dict,
    config: dict,
) -> Optional[dict]:
    """Interactive colour-mapping dashboard.

    Parameters
    ----------
    closest_indices_img : np.ndarray
        Per-pixel palette index image (H×W, int).
    fg_alpha : np.ndarray
        Foreground alpha mask (H×W, uint8).
    present_indices : list[int]
        Ordered list of palette indices present in the image.
    layer_modes : dict
        ``{palette_idx: 'fill' | 'outline'}`` for each layer.
    config : dict
        Full application config.

    Returns
    -------
    dict or None
        On confirm: ``{present_color_indices, mapped_color_indices,
        layer_modes, bg_color_idx}``.  ``None`` on cancel.
    """
    # Mutable state
    state = {
        "selected": 0,         # int index or "bg"
        "present": list(present_indices),
        "mapped": list(range(len(COLORS_PALETTE))),
        "modes": dict(layer_modes),
        "scroll": 0,
        "bg_idx": -1,
    }

    def _handle_row_click(idx: int, mx: int) -> None:
        orig = state["present"][idx]
        if 565 <= mx <= 635:
            state["modes"][orig] = "outline" if state["modes"][orig] == "fill" else "fill"
            print(f"Toggled layer {idx + 1} to {state['modes'][orig]}")
        elif 655 <= mx <= 685:
            if idx > 0:
                p = state["present"]
                p[idx], p[idx - 1] = p[idx - 1], p[idx]
                state["selected"] = idx - 1
                print(f"Moved layer {idx + 1} UP")
        elif 695 <= mx <= 725:
            if idx < len(state["present"]) - 1:
                p = state["present"]
                p[idx], p[idx + 1] = p[idx + 1], p[idx]
                state["selected"] = idx + 1
                print(f"Moved layer {idx + 1} DOWN")
        else:
            state["selected"] = idx
            print(f"Selected layer {idx + 1}")

    def _on_mouse(event: int, mx: int, my: int, flags: int, param: object) -> None:
        N = len(state["present"])

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Middle panel — layers list
        if 480 <= mx <= 720:
            # Background row
            if 80 <= my <= 118:
                state["selected"] = "bg"
                print("Selected BACKGROUND layer")
                return

            if N <= 9:
                for i in range(N):
                    y_min = 125 + i * 46
                    y_max = y_min + 38
                    if y_min <= my <= y_max:
                        _handle_row_click(i, mx)
                        return
            else:
                if 125 <= my <= 150:
                    state["scroll"] = max(0, state["scroll"] - 1)
                    return
                if 535 <= my <= 560:
                    state["scroll"] = min(N - 8, state["scroll"] + 1)
                    return
                for i in range(8):
                    y_min = 160 + i * 46
                    y_max = y_min + 38
                    if y_min <= my <= y_max:
                        _handle_row_click(i + state["scroll"], mx)
                        return

        # Palette grid click
        for idx, item in enumerate(COLORS_PALETTE[:22]):
            col_i = idx % 4
            row_i = idx // 4
            cx = 745 + col_i * 45 + 22
            cy = 85 + row_i * 50 + 22
            if np.sqrt((mx - cx) ** 2 + (my - cy) ** 2) <= 18:
                sel = state["selected"]
                if sel == "bg":
                    state["bg_idx"] = idx
                    print(f"Mapped BACKGROUND to '{item[3]}'")
                elif isinstance(sel, int) and sel < len(state["present"]):
                    state["mapped"][state["present"][sel]] = idx
                    print(f"Mapped layer {sel + 1} to '{item[3]}'")
                return

        # Toggle Skip button
        if 480 <= mx <= 720 and 580 <= my <= 615:
            sel = state["selected"]
            if sel == "bg":
                state["bg_idx"] = 0 if state["bg_idx"] == -1 else -1
                print(f"Toggled BACKGROUND to {'ON' if state['bg_idx'] != -1 else 'OFF'}")
            elif isinstance(sel, int) and sel < len(state["present"]):
                orig = state["present"][sel]
                state["mapped"][orig] = orig if state["mapped"][orig] == -1 else -1
                print(f"Toggled skip for layer {sel + 1}")

    window = "Color Mapping Panel (ENTER to Confirm, ESC to Cancel)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)

    print("\n--- MAPPING DASHBOARD ---")
    print("  Click layers to select | Click palette to remap colours")
    print("  ENTER/SPACE to Start Drawing | ESC to Cancel")

    while True:
        preview = get_preview_image(
            closest_indices_img, fg_alpha, state["present"],
            state["mapped"], state["modes"], state["selected"],
            state["bg_idx"],
        )
        canvas = _draw_dashboard_canvas(
            preview, state["present"], state["selected"],
            state["mapped"], state["modes"], state["scroll"],
            state["bg_idx"],
        )

        cv2.imshow(window, canvas)

        key = cv2.waitKey(15) & 0xFF
        if key in (13, 32):
            cv2.destroyAllWindows()
            return {
                "present_color_indices": state["present"],
                "mapped_color_indices": state["mapped"],
                "layer_modes": state["modes"],
                "bg_color_idx": state["bg_idx"],
            }
        elif key in (27, ord('q'), ord('c')):
            print("Dashboard cancelled by user.")
            cv2.destroyAllWindows()
            return None
