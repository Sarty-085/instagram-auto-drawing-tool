"""calibrate_slider — Interactive slider position calibration utility.

Takes a screenshot of the phone and lets the user click on points
to identify the brush slider coordinates.  Results are saved to
``config.json``.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from adb_utils import ADBConnection
from config import load_config, save_config


def main() -> None:
    """Run the slider calibration workflow."""
    try:
        adb = ADBConnection()
    except (FileNotFoundError, ConnectionError) as exc:
        print(f"❌ {exc}")
        return

    config = load_config()

    print("\n--- SLIDER CALIBRATION ---")
    print("1. Open Instagram Draw mode on your phone.")
    print("2. Make the brush slider visible on screen.")
    print("3. Press ENTER when ready...")
    input()

    screenshot = adb.screenshot()
    h, w = screenshot.shape[:2]

    scale = min(1.0, 800 / h)
    display = cv2.resize(screenshot, (int(w * scale), int(h * scale)))

    clicks: List[Tuple[int, int]] = []

    def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            px, py = int(x / scale), int(y / scale)
            clicks.append((px, py))
            label = f"({px}, {py})"
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display, label, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("Slider Calibration", display)
            print(f"  Click {len(clicks)}: {label}")

    window = "Slider Calibration"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)

    print("\nClick the TOP of the brush slider (thickest),")
    print("then the BOTTOM (thinnest). Press any key after both clicks.")

    cv2.imshow(window, display)

    while len(clicks) < 2:
        if cv2.waitKey(100) == 27:
            print("Cancelled.")
            cv2.destroyAllWindows()
            return

    cv2.waitKey(1500)
    cv2.destroyAllWindows()

    thick_x, thick_y = clicks[0]
    _thin_x, thin_y = clicks[1]

    brush_x = thick_x
    top_y = min(thick_y, thin_y)
    bot_y = max(thick_y, thin_y)

    step = (bot_y - top_y) / 4
    for i in range(5):
        level = 5 - i
        y_pos = int(top_y + i * step)
        config["brush_config"][str(level)]["y"] = y_pos

    config["device"]["brush_slider_x"] = brush_x
    save_config(config)

    print("\n" + "=" * 50)
    print("✅ SLIDER CALIBRATION COMPLETE — saved to config.json")
    print("=" * 50)
    print(f"  Slider X: {brush_x}")
    for i in range(5):
        level = 5 - i
        print(f"  Size {level}: Y={config['brush_config'][str(level)]['y']}")


if __name__ == "__main__":
    main()
