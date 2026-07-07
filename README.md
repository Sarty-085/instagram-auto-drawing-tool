# Instagram Stories Auto-Drawing Bot

An open-source desktop utility that automatically color-maps, scales, overlays, and draws images onto the Instagram Stories drawing canvas on an Android phone using Android Debug Bridge (ADB).

---

## 🌟 Key Features

*   **Interactive Visual Overlay editor**: Overlay any transparent image directly on top of your phone screen, drag to position, and drag the bottom-right corner to scale to fit.
*   **Active Layer Highlights**: Selecting a layer in the dashboard highlights it in the Live Preview while dimming other layers, letting you inspect exactly what is drawn on that layer.
*   **Layer Manager & Reordering**: Easily scroll through color layers, toggle visibility, switch mode between solid fills (`FILL`) or outline tracing (`LINE`), and reorder drawing sequences so background passes run first.
*   **Strict Mode Separation**:
    *   `FILL` mode: Draws tightly spaced scanlines (`step_size = 6px`) to ensure solid fills with zero gaps.
    *   `LINE` mode: Traces high-fidelity outer contours.
*   **Safe Touch-Down Redirection**: Automatically rolls start coordinates of swipe gestures away from the left edge (`X >= 120`). This completely prevents the Instagram brush slider or Android system navigation from triggering during drawing.
*   **Settle Pauses & Swipe Drags**: Uses horizontal drags to change brush sizes on the slider reliably, with post-background settle delays to clear Android gesture queues.
*   **Dynamic Drawing Speeds**: Short details/outlines are drawn quickly at `120ms` per segment, while long fills or backgrounds draw slowly at `800ms` for maximum precision and screen registration.
*   **Ultra-Smooth Curves**: Features a customizable curve approximation threshold (`CONTOUR_EPSILON = 0.0012`) to break curved outlines into short segments, making loops, circles, and curves draw smooth and round.

---

## 🚀 How to Run (Precompiled Executable)

If you compile the script into a standalone app (`draw_interactive.exe`), users can run it without installing Python!

1.  **Enable Android USB Debugging**:
    *   On your Android phone, go to **Settings** > **About Phone**.
    *   Tap **Build Number** 7 times to unlock Developer Options.
    *   Go to **Developer Options** and enable **USB Debugging**.
2.  **Connect Phone**:
    *   Connect your phone to your PC via USB.
    *   On your phone, allow the USB Debugging permission prompt.
3.  **Run the App**:
    *   Ensure `adb.exe`, `AdbWinApi.dll`, and `AdbWinUsbApi.dll` are in the same folder as `draw_interactive.exe`.
    *   Double-click `draw_interactive.exe`.
    *   Follow the screen prompts to position your drawing and map colors!

---

## 🛠️ How to Run from Source (Python)

If you prefer to run the project directly from Python:

### 1. Prerequisites
Install the required packages in your Python environment:
```powershell
pip install opencv-python numpy
```

### 2. Run the Script
Open your terminal inside the project directory and run:
```powershell
python draw_interactive.py
```

---

## 📦 How to Package into a Standalone `.exe`

You can package this project into a single, portable Windows application using **PyInstaller**. This bundles Python, OpenCV, NumPy, and the ADB dependencies into a single double-clickable `.exe` file.

1.  **Install PyInstaller**:
    ```powershell
    python -m pip install pyinstaller
    ```
2.  **Compile the Executable**:
    Run the following command in the directory containing `draw_interactive.py` and the `adb.exe` files:
    ```powershell
    python -m PyInstaller --onefile --add-binary "adb.exe;." --add-binary "AdbWinApi.dll;." --add-binary "AdbWinUsbApi.dll;." draw_interactive.py
    ```
3.  **Find the App**:
    The compiled standalone executable will be created inside the **`dist`** folder. Copy `draw_interactive.exe` out and share it with anyone!

---

## ⚙️ Calibration Settings
At the top of `draw_interactive.py`, you can fine-tune several configurations for your specific device:
*   `BRUSH_X`: The horizontal coordinate of the Instagram brush slider.
*   `BRUSH_CONFIG`: A dictionary mapping brush size numbers (1 to 5) to their slider Y coordinates and actual drawn pixel widths. Run the `scratch/measure_brush_sizes.py` script on your phone to calibrate this!
*   `FILL_STEP_SIZE`: The horizontal distance between scanlines in solid fills (defaults to `6px` for solid fills).
*   `CONTOUR_EPSILON`: Epsilon factor for curve smoothing. Set to `0.0012` for ultra-smooth curves, or higher (e.g. `0.005`) for faster, simplified outlines.

---

## 📄 License
This project is open-sourced under the MIT License. Feel free to use, modify, and distribute it!
