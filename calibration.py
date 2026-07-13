"""calibration — Auto + manual calibration system for the Instagram auto-drawing bot.

Provides the :class:`DeviceCalibration` class which can:

* **Auto-calibrate** by taking a screenshot, detecting the colour-palette bar
  with Hough circle detection, and mapping individual colour positions.
* **Manually calibrate** the brush-size slider via an interactive OpenCV
  window where the user clicks the thickest and thinnest points.
* **Run a full calibration flow** that chains auto-detection, optional manual
  override, and config persistence.

All coordinate detection is device-agnostic — results are derived from the
live screenshot rather than hard-coded pixel values.
"""

from __future__ import annotations

import copy
import subprocess
import time
from typing import List, Tuple

import cv2
import numpy as np

from adb_utils import ADBConnection
from config import DEFAULT_CONFIG, load_config, save_config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_COLOR_X_POSITIONS: List[int] = [432, 515, 597, 680, 763, 845, 928]
"""Fallback horizontal positions (px) of the palette colour circles when
auto-detection fails.  Based on a 1080-wide display."""

_PALETTE_SCAN_RATIO: float = 0.30
"""Fraction of the screen (from the bottom) to scan when searching for the
colour-palette bar."""

_HOUGH_CIRCLE_MIN_RADIUS: int = 15
_HOUGH_CIRCLE_MAX_RADIUS: int = 40

_PALETTE_STRIP_HALF_HEIGHT: int = 30
"""Vertical padding (px) above and below ``palette_y`` when extracting the
horizontal strip used for colour-circle detection."""

_MAX_DISPLAY_HEIGHT: int = 800
"""Maximum pixel height of the OpenCV preview window used in manual
calibration."""

_NUM_BRUSH_SIZES: int = 5
"""Number of equidistant brush-size stops calculated between the thickest
and thinnest slider endpoints."""


class DeviceCalibration:
    """Detects and stores device-specific drawing coordinates.

    Parameters
    ----------
    adb : ADBConnection
        An already-verified ADB connection to the target device.
    """

    def __init__(self, adb: ADBConnection) -> None:
        """Initialise the calibration helper.

        Stores the ADB connection and queries the device's physical screen
        size so that all subsequent detection can be resolution-aware.

        Parameters
        ----------
        adb : ADBConnection
            A connected ADB instance.
        """
        self.adb: ADBConnection = adb
        self.screen_width: int
        self.screen_height: int
        self.screen_width, self.screen_height = adb.get_screen_size()
        print(
            f"[CALIBRATION] Screen size: "
            f"{self.screen_width}x{self.screen_height}"
        )

    # ------------------------------------------------------------------
    # Auto-calibration
    # ------------------------------------------------------------------

    def auto_calibrate(self) -> dict:
        """Attempt to auto-detect device-specific coordinates.

        The method takes a screenshot, searches for the colour-palette bar
        and individual colour circles using Hough circle detection, and
        assembles a configuration dictionary built on top of
        ``DEFAULT_CONFIG``.

        Returns
        -------
        dict
            A complete configuration dictionary with detected values
            merged into the defaults.
        """
        print("\n[AUTO-CALIBRATE] Taking screenshot …")
        screenshot: np.ndarray = self.adb.screenshot()

        palette_y: int = self._detect_palette_y(screenshot)
        color_positions: List[int] = self._detect_color_positions(
            screenshot, palette_y
        )

        # Build config on top of defaults
        config: dict = copy.deepcopy(DEFAULT_CONFIG)
        config["device"]["screen_width"] = self.screen_width
        config["device"]["screen_height"] = self.screen_height
        config["device"]["palette_y"] = palette_y
        config["device"]["palette_x_positions"] = color_positions

        print("\n[AUTO-CALIBRATE] Detection summary:")
        print(f"  Screen size       : {self.screen_width}x{self.screen_height}")
        print(f"  Palette Y         : {palette_y}")
        print(f"  Colour X positions: {color_positions}")

        print("\n[AUTO-CALIBRATE] Calibrating custom color spectrum picker...")
        print("  Make sure Instagram Draw is open on Page 1...")
        spectrum_info = self.calibrate_spectrum(config)
        if spectrum_info:
            config.update(spectrum_info)
            print("  Custom color spectrum calibrated successfully!")
        else:
            print("  [WARNING] Custom color spectrum calibration failed or skipped.")

        return config

    def calibrate_spectrum(self, config: dict) -> dict:
        """Auto-detect the horizontal color spectrum picker by sending a long-press.

        Requires Instagram Draw to be open on Page 1.
        """
        device_cfg = config.get("device", {})
        palette_y = device_cfg.get("palette_y")
        palette_x_positions = device_cfg.get("palette_x_positions")

        if not palette_y or not palette_x_positions:
            return {}

        swatch_x = palette_x_positions[0] # White swatch

        # Run non-blocking adb touch-and-hold (long-press) in background
        adb_cmd = [
            self.adb.adb_path,
            "-s", self.adb.device_serial,
            "shell", "input", "swipe",
            str(swatch_x), str(palette_y),
            str(swatch_x), str(palette_y),
            "5000"
        ]
        try:
            proc = subprocess.Popen(adb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"  [SPECTRUM] Error starting long-press: {e}")
            return {}

        # Wait for spectrum to appear
        time.sleep(1.5)

        # Capture screenshot
        screenshot = self.adb.screenshot()

        # Wait for swipe to finish
        proc.wait()

        # Scan for rainbow spectrum row
        h, w, c = screenshot.shape
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        sats = hsv[:, :, 1]
        vals = hsv[:, :, 2]
        hues = hsv[:, :, 0]

        best_row = -1
        max_score = 0

        # Scan rows above palette_y
        scan_start = max(0, palette_y - 120)
        scan_end = max(0, palette_y - 10)

        for row in range(scan_start, scan_end):
            mask = (sats[row] > 80) & (vals[row] > 100)
            colorful_count = np.sum(mask)

            if colorful_count > w * 0.3:
                valid_hues = hues[row, mask]
                hue_std = np.std(valid_hues)
                unique_bins = len(np.unique(valid_hues // 10))

                if hue_std > 15 and unique_bins >= 8:
                    score = colorful_count * hue_std
                    if score > max_score:
                        max_score = score
                        best_row = row

        if best_row != -1:
            mask_best = (sats[best_row] > 80) & (vals[best_row] > 100)
            colorful_indices = np.where(mask_best)[0]
            start_x = int(colorful_indices[0])
            end_x = int(colorful_indices[-1])

            # Extract BGR values along the row
            colors = []
            for x in range(start_x, end_x + 1):
                colors.append(screenshot[best_row, x].tolist())

            return {
                "spectrum": {
                    "y": best_row,
                    "start_x": start_x,
                    "end_x": end_x,
                    "colors": colors
                }
            }

        return {}

    def _detect_palette_y(self, screenshot: np.ndarray) -> int:
        """Detect the vertical position of the colour-palette bar.

        Strategy
        --------
        1. Crop the bottom ``_PALETTE_SCAN_RATIO`` of the screenshot.
        2. Convert to grayscale and threshold to isolate bright regions.
        3. Run :func:`cv2.HoughCircles` to find circles with radii in
           ``[_HOUGH_CIRCLE_MIN_RADIUS, _HOUGH_CIRCLE_MAX_RADIUS]``.
        4. If circles are found, return the **median Y** (in full-image
           coordinates).
        5. Otherwise, return a heuristic fallback at 91.5 % of screen height.

        Parameters
        ----------
        screenshot : np.ndarray
            BGR screenshot captured from the device.

        Returns
        -------
        int
            Estimated Y coordinate of the palette bar.
        """
        img_h, img_w = screenshot.shape[:2]
        crop_start: int = int(img_h * (1.0 - _PALETTE_SCAN_RATIO))
        bottom_region: np.ndarray = screenshot[crop_start:, :]

        gray: np.ndarray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Slight blur to reduce noise before Hough detection
        blurred: np.ndarray = cv2.GaussianBlur(thresh, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=_HOUGH_CIRCLE_MIN_RADIUS,
            maxRadius=_HOUGH_CIRCLE_MAX_RADIUS,
        )

        if circles is not None:
            detected: np.ndarray = np.round(circles[0]).astype(int)
            # Y values are relative to the cropped region — offset them.
            y_values: np.ndarray = detected[:, 1] + crop_start
            palette_y: int = int(np.median(y_values))
            print(
                f"[DETECT] Palette Y detected via HoughCircles: {palette_y} "
                f"({len(detected)} circle(s) found)"
            )
            return palette_y

        # Fallback heuristic
        fallback_y: int = int(self.screen_height * 0.915)
        print(
            f"[DETECT] HoughCircles did not find palette circles. "
            f"Using fallback palette Y: {fallback_y}"
        )
        return fallback_y

    def _detect_color_positions(
        self, screenshot: np.ndarray, palette_y: int
    ) -> List[int]:
        """Detect the horizontal positions of individual colour circles.

        A narrow horizontal strip centred on ``palette_y`` is extracted and
        cropped horizontally to keep only the area containing the 7 color swatches,
        excluding the close, undo, dropper, and send buttons. The cropped region
        is searched with :func:`cv2.HoughCircles`.

        Parameters
        ----------
        screenshot : np.ndarray
            BGR screenshot captured from the device.
        palette_y : int
            Vertical centre of the palette bar.

        Returns
        -------
        list[int]
            Sorted list of X pixel coordinates for each colour circle.
            Falls back to scaled ``_DEFAULT_COLOR_X_POSITIONS`` if detection fails or is incomplete.
        """
        img_h, img_w = screenshot.shape[:2]

        # Horizontal cropping bounds: 38% to 88% of screen width contains exactly the 7 color circles.
        crop_left: int = int(img_w * 0.38)
        crop_right: int = int(img_w * 0.88)

        y_top: int = max(0, palette_y - _PALETTE_STRIP_HALF_HEIGHT)
        y_bot: int = min(img_h, palette_y + _PALETTE_STRIP_HALF_HEIGHT)

        # Crop the strip horizontally to exclude close/undo/dropper on left and send on right
        strip: np.ndarray = screenshot[y_top:y_bot, crop_left:crop_right]

        gray: np.ndarray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        blurred: np.ndarray = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=_HOUGH_CIRCLE_MIN_RADIUS,
            maxRadius=_HOUGH_CIRCLE_MAX_RADIUS,
        )

        scaled_defaults = [int(x * img_w / 1080) for x in _DEFAULT_COLOR_X_POSITIONS]

        if circles is not None:
            detected: np.ndarray = np.round(circles[0]).astype(int)
            # Add crop_left to convert coordinates back to full screen space
            x_coords: List[int] = sorted(int(c[0] + crop_left) for c in detected)

            # We expect exactly 7 color circles
            if len(x_coords) == 7:
                print(
                    f"[DETECT] Successfully detected all 7 colour circle(s) at X: "
                    f"{x_coords}"
                )
                return x_coords
            else:
                print(
                    f"[DETECT] Found {len(x_coords)} circles in the color swatch area (expected 7). "
                    f"Falling back to scaled defaults: {scaled_defaults}"
                )
                return scaled_defaults

        print(
            "[DETECT] Could not detect colour circles. "
            f"Using scaled defaults: {scaled_defaults}"
        )
        return scaled_defaults

    # ------------------------------------------------------------------
    # Manual slider calibration
    # ------------------------------------------------------------------

    def manual_calibrate_slider(self) -> dict:
        """Interactively calibrate the brush-size slider via mouse clicks.

        Displays a scaled screenshot in an OpenCV window and asks the user
        to click:

        1. The **top** of the brush slider (thickest brush).
        2. The **bottom** of the brush slider (thinnest brush).

        Five equidistant Y positions are computed between those two points
        and returned as a partial config dictionary.

        Returns
        -------
        dict
            Partial configuration containing ``brush_slider_x`` under
            ``"device"`` and five brush entries under ``"brush_config"``.
        """
        print("\n[MANUAL CALIBRATION] Taking screenshot …")
        screenshot: np.ndarray = self.adb.screenshot()
        img_h, img_w = screenshot.shape[:2]

        # Scale down for display if necessary
        scale: float = min(1.0, _MAX_DISPLAY_HEIGHT / img_h)
        display_img: np.ndarray = cv2.resize(
            screenshot, (int(img_w * scale), int(img_h * scale))
        )

        clicks: List[Tuple[int, int]] = []

        def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
            """Record mouse click positions, mapped back to device coords."""
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
                # Map display coordinates back to full-resolution coords
                orig_x: int = int(x / scale)
                orig_y: int = int(y / scale)
                clicks.append((orig_x, orig_y))

                label: str = "THICK (top)" if len(clicks) == 1 else "THIN (bottom)"
                print(f"  Click {len(clicks)}: ({orig_x}, {orig_y}) — {label}")

                # Draw a marker on the display image
                cv2.circle(display_img, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(
                    display_img,
                    label,
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                cv2.imshow("Brush Slider Calibration", display_img)

        print(
            "Click the TOP of the brush slider (thickest), "
            "then the BOTTOM (thinnest)."
        )
        print("Press any key after both clicks to continue.")

        window_name: str = "Brush Slider Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, _on_mouse)
        cv2.imshow(window_name, display_img)

        # Wait until 2 clicks are registered
        while len(clicks) < 2:
            key: int = cv2.waitKey(100)
            if key == 27:  # ESC to abort
                print("[MANUAL CALIBRATION] Cancelled by user.")
                cv2.destroyWindow(window_name)
                return {}

        # Allow user a moment to see the markers, then close
        cv2.waitKey(1500)
        cv2.destroyWindow(window_name)

        thick_x, thick_y = clicks[0]
        _thin_x, thin_y = clicks[1]

        # Use the X from the first click as the slider X
        brush_slider_x: int = thick_x

        # Ensure thick_y < thin_y (top is smaller Y)
        top_y: int = min(thick_y, thin_y)
        bot_y: int = max(thick_y, thin_y)

        # Calculate equidistant stops (1 = thinnest, 5 = thickest)
        step: float = (bot_y - top_y) / (_NUM_BRUSH_SIZES - 1)
        brush_config: dict = {}
        for i in range(_NUM_BRUSH_SIZES):
            level: int = _NUM_BRUSH_SIZES - i  # 5 (thickest) → 1 (thinnest)
            y_pos: int = int(top_y + i * step)
            # Approximate pixel width — linearly interpolated
            width: int = int(114 - (114 - 27) * (i / (_NUM_BRUSH_SIZES - 1)))
            brush_config[str(level)] = {"y": y_pos, "width": width}

        print("\n[MANUAL CALIBRATION] Brush slider results:")
        print(f"  Slider X  : {brush_slider_x}")
        for level, vals in sorted(brush_config.items(), reverse=True):
            print(f"  Brush {level}   : Y={vals['y']}, width≈{vals['width']}px")

        return {
            "device": {"brush_slider_x": brush_slider_x},
            "brush_config": brush_config,
        }

    # ------------------------------------------------------------------
    # Full calibration flow
    # ------------------------------------------------------------------

    def run_calibration(self) -> dict:
        """Run the complete calibration workflow.

        Steps
        -----
        1. Print a welcome banner.
        2. Run :meth:`auto_calibrate` to detect screen and palette values.
        3. Ask the user whether auto-detection looks correct.
        4. If the user rejects the results, run :meth:`manual_calibrate_slider`
           and merge the manual overrides into the auto-detected config.
        5. Save the final configuration via :func:`config.save_config`.
        6. Return the saved config.

        Returns
        -------
        dict
            The finalised, persisted configuration dictionary.
        """
        print("=" * 60)
        print("   INSTAGRAM AUTO-DRAWING BOT — DEVICE CALIBRATION")
        print("=" * 60)
        print(
            "This wizard will auto-detect your device's drawing\n"
            "coordinates and optionally let you fine-tune the\n"
            "brush-size slider manually.\n"
        )

        config: dict = self.auto_calibrate()

        print("\n" + "-" * 60)
        answer: str = (
            input(
                "Does the auto-detection look correct? [Y/n]: "
            )
            .strip()
            .lower()
        )

        if answer in ("n", "no"):
            manual_overrides: dict = self.manual_calibrate_slider()
            if manual_overrides:
                # Deep-merge manual results on top of auto-detected config
                for section, values in manual_overrides.items():
                    if section in config and isinstance(config[section], dict):
                        config[section].update(values)
                    else:
                        config[section] = values
                print("\n[CALIBRATION] Manual overrides merged.")

        save_config(config)

        print("\n[CALIBRATION] Calibration complete!")
        print("=" * 60)
        return config
