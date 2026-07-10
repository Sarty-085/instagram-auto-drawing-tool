"""measure_brush_sizes — Brush diameter calibration utility.

Draws 5 test dots on the Instagram canvas (one per brush size) and
guides the user through clicking the edges of each dot to measure
the pixel diameter.  Results are saved to ``config.json`` via the
shared :mod:`config` module.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np

from adb_utils import ADBConnection
from config import load_config, save_config


def main() -> None:
    """Run the brush-size measurement workflow."""
    # Connect
    try:
        adb = ADBConnection()
    except (FileNotFoundError, ConnectionError) as exc:
        print(f"❌ {exc}")
        return

    config = load_config()
    brush_cfg = config["brush_config"]
    brush_x: int = int(config["device"]["brush_slider_x"])

    slider_y_coords = [int(brush_cfg[str(i)]["y"]) for i in range(1, 6)]
    draw_y_coords = [400, 750, 1100, 1450, 1800]

    print("\n--- BRUSH SIZE MEASUREMENT ---")
    print("1. Open Instagram Draw mode on your phone.")
    print("2. Choose a clean canvas (solid dark background).")
    print("3. Select WHITE colour from the palette.")
    print("4. Press ENTER when ready...")
    input()

    # Draw 5 calibration dots
    print("Drawing calibration dots...")
    for idx, slider_y in enumerate(slider_y_coords):
        print(f"  Dot {idx + 1}/5: Slider Y={slider_y}, Draw Y={draw_y_coords[idx]}")
        adb.swipe(brush_x, slider_y, brush_x + 15, slider_y, 200)
        time.sleep(0.5)
        adb.tap(500, draw_y_coords[idx])
        time.sleep(0.5)

    print("Taking screenshot...")
    time.sleep(1.0)
    screenshot = adb.screenshot()
    h_screen, w_screen = screenshot.shape[:2]

    scale = min(1.0, 800 / h_screen)
    disp_w, disp_h = int(w_screen * scale), int(h_screen * scale)
    display_img = cv2.resize(screenshot, (disp_w, disp_h))

    click_points: List[Tuple[int, int]] = []

    def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            px, py = int(x / scale), int(y / scale)
            click_points.append((px, py))
            print(f"  Clicked: ({px}, {py})")

    window = "Brush Diameter Calibration"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)

    measured: List[int] = []

    print("\n--- MEASUREMENT ---")
    print("Click LEFT edge then RIGHT edge of each dot (top to bottom).\n")

    for idx, draw_y in enumerate(draw_y_coords):
        size_num = idx + 1
        print(f"[Size {size_num}] Click LEFT edge of dot...")
        click_points.clear()

        while len(click_points) < 1:
            canvas = display_img.copy()
            dy = int(draw_y * scale)
            cv2.line(canvas, (0, dy), (disp_w, dy), (0, 0, 255), 1)
            cv2.imshow(window, canvas)
            if cv2.waitKey(30) & 0xFF == 27:
                cv2.destroyAllWindows()
                return

        pt_left = click_points[0]
        cv2.drawMarker(display_img, (int(pt_left[0] * scale), int(pt_left[1] * scale)),
                       (0, 255, 0), cv2.MARKER_CROSS, 8, 1)

        print(f"[Size {size_num}] Click RIGHT edge of same dot...")
        while len(click_points) < 2:
            cv2.imshow(window, display_img.copy())
            if cv2.waitKey(30) & 0xFF == 27:
                cv2.destroyAllWindows()
                return

        pt_right = click_points[1]
        width = abs(pt_right[0] - pt_left[0])
        measured.append(width)
        print(f"  → Size {size_num} diameter: {width}px")

        # Visual feedback
        cx = int((pt_left[0] + pt_right[0]) / 2 * scale)
        cy = int(draw_y * scale)
        r = int(width / 2 * scale)
        cv2.circle(display_img, (cx, cy), r, (255, 255, 0), 1)

    cv2.destroyAllWindows()

    # Update config
    for i in range(5):
        config["brush_config"][str(i + 1)]["width"] = measured[i]

    save_config(config)

    print("\n" + "=" * 50)
    print("✅ CALIBRATION COMPLETE — saved to config.json")
    print("=" * 50)
    for i in range(5):
        print(f"  Size {i + 1}: Y={slider_y_coords[i]}, width={measured[i]}px")


if __name__ == "__main__":
    main()
