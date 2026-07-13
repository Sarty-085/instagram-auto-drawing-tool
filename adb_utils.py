"""adb_utils — Shared ADB communication module for the Instagram auto-drawing bot.

Provides the :class:`ADBConnection` class which handles:
* Portable ADB binary discovery (script dir → cwd → system PATH).
* Device connection verification.
* Screen capture → NumPy BGR array.
* Touch input (tap / swipe).
* Screen-size querying.

Every subprocess call uses **list arguments** (never ``shell=True``) and
checks the return code, raising a clear exception on failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Tuple

import cv2
import numpy as np


class ADBConnection:
    """Manages a validated connection to an Android device via ADB.

    On instantiation the class locates the ``adb`` binary, verifies that
    exactly one device is connected, and exposes convenience methods for
    screenshots, taps, swipes, and display-size queries.

    Raises
    ------
    FileNotFoundError
        If the ``adb`` executable cannot be found anywhere.
    ConnectionError
        If no Android device is detected by ``adb devices``.
    """

    def __init__(self) -> None:
        """Initialise the ADB connection.

        Calls :meth:`_find_adb` to locate the binary, then
        :meth:`_verify_connection` to ensure a device is available.
        """
        self.adb_path: str = self._find_adb()
        print(f"Using ADB: {self.adb_path}")
        self._verify_connection()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_adb(self) -> str:
        """Locate the ``adb`` executable using a portable search order.

        Search order
        ------------
        1. Directory containing the running script (or frozen ``.exe``).
        2. Current working directory.
        3. System ``PATH`` (via :func:`shutil.which`).

        Returns
        -------
        str
            Absolute path to the ``adb`` executable.

        Raises
        ------
        FileNotFoundError
            If ``adb`` is not found in any of the searched locations.
        """
        adb_name = "adb.exe" if sys.platform == "win32" else "adb"

        # 1. Directory of the script / frozen exe
        if getattr(sys, "frozen", False):
            script_dir = os.path.dirname(sys.executable)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))

        candidate = os.path.join(script_dir, adb_name)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

        # 2. Current working directory
        candidate = os.path.join(os.getcwd(), adb_name)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

        # 3. System PATH
        found = shutil.which("adb")
        if found is not None:
            return os.path.abspath(found)

        raise FileNotFoundError(
            "Could not find the 'adb' executable. "
            "Place it next to this script, in the current working directory, "
            "or ensure it is available on your system PATH."
        )

    def _verify_connection(self) -> None:
        """Verify that at least one Android device is connected.

        Runs ``adb devices`` and parses the output for lines whose second
        column is ``device``.  Prints the serial number of the first
        connected device.

        Raises
        ------
        ConnectionError
            If no device with status ``device`` is found.
        RuntimeError
            If the ``adb devices`` command itself fails.
        """
        result = subprocess.run(
            [self.adb_path, "devices"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"'adb devices' failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )

        devices: list[str] = []
        for line in result.stdout.strip().splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])

        if not devices:
            raise ConnectionError(
                "No Android device found. "
                "Connect a device via USB (with USB-debugging enabled) or "
                "start an emulator, then try again."
            )

        self.device_serial: str = devices[0]
        print(f"Connected device: {self.device_serial}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def screenshot(self) -> np.ndarray:
        """Capture the device screen and return it as a BGR NumPy array.

        Runs ``adb exec-out screencap -p`` to stream raw PNG bytes, then
        decodes them with OpenCV.

        Returns
        -------
        numpy.ndarray
            The screenshot as a BGR image (H × W × 3).

        Raises
        ------
        RuntimeError
            If the screencap command fails or the image cannot be decoded.
        """
        result = subprocess.run(
            [self.adb_path, "exec-out", "screencap", "-p"],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"'adb exec-out screencap -p' failed (rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )

        png_bytes = result.stdout
        if not png_bytes:
            raise RuntimeError("screencap returned empty output.")

        img_array = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise RuntimeError(
                "Failed to decode the screenshot PNG data. "
                f"Received {len(png_bytes)} bytes from screencap."
            )

        return img

    def tap(self, x: int, y: int, delay: float = 0.5) -> None:
        """Send a tap event to the device at (x, y).

        Parameters
        ----------
        x : int
            Horizontal coordinate (pixels).
        y : int
            Vertical coordinate (pixels).
        delay : float, optional
            Seconds to sleep after the tap (default ``0.5``).

        Raises
        ------
        RuntimeError
            If the ``adb shell input tap`` command fails.
        """
        result = subprocess.run(
            [self.adb_path, "shell", "input", "tap", str(x), str(y)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"'adb shell input tap {x} {y}' failed "
                f"(rc={result.returncode}): {result.stderr.strip()}"
            )
        time.sleep(delay)

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: int = 50,
    ) -> None:
        """Send a swipe gesture from (x1, y1) to (x2, y2).

        Parameters
        ----------
        x1, y1 : int
            Start coordinates (pixels).
        x2, y2 : int
            End coordinates (pixels).
        duration : int, optional
            Duration of the swipe in milliseconds (default ``50``).

        Raises
        ------
        RuntimeError
            If the ``adb shell input swipe`` command fails.
        """
        result = subprocess.run(
            [
                self.adb_path, "shell", "input", "swipe",
                str(x1), str(y1), str(x2), str(y2), str(duration),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"'adb shell input swipe {x1} {y1} {x2} {y2} {duration}' "
                f"failed (rc={result.returncode}): {result.stderr.strip()}"
            )

    def drag_and_drop(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: int = 1500,
    ) -> None:
        """Send a drag-and-drop gesture from (x1, y1) to (x2, y2).

        This starts with a touch down and hold (long press) at (x1, y1),
        moves to (x2, y2), and touches up.

        Parameters
        ----------
        x1, y1 : int
            Start coordinates (pixels).
        x2, y2 : int
            End coordinates (pixels).
        duration : int, optional
            Duration of the drag gesture in milliseconds (default ``1500``).

        Raises
        ------
        RuntimeError
            If the ``adb shell input draganddrop`` command fails.
        """
        result = subprocess.run(
            [
                self.adb_path, "shell", "input", "draganddrop",
                str(x1), str(y1), str(x2), str(y2), str(duration),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"'adb shell input draganddrop {x1} {y1} {x2} {y2} {duration}' "
                f"failed (rc={result.returncode}): {result.stderr.strip()}"
            )

    def get_screen_size(self) -> Tuple[int, int]:
        """Query the physical display size of the connected device.

        Runs ``adb shell wm size`` and parses the ``Physical size: WxH``
        line from the output.

        Returns
        -------
        tuple[int, int]
            ``(width, height)`` in pixels.

        Raises
        ------
        RuntimeError
            If the command fails or the output cannot be parsed.
        """
        result = subprocess.run(
            [self.adb_path, "shell", "wm", "size"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"'adb shell wm size' failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )

        for line in result.stdout.strip().splitlines():
            if "Physical size:" in line:
                # Expected format: "Physical size: 1080x1920"
                size_str = line.split(":")[-1].strip()
                try:
                    w, h = size_str.split("x")
                    return int(w), int(h)
                except ValueError:
                    raise RuntimeError(
                        f"Could not parse screen dimensions from '{size_str}'."
                    )

        raise RuntimeError(
            f"'Physical size:' line not found in wm size output:\n"
            f"{result.stdout.strip()}"
        )
