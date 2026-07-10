# 🎨 Instagram Auto-Drawing Tool

> Automatically draw any image on Instagram's in-app drawing canvas using ADB — pixel-perfect, colour-mapped, and fully interactive.

---

## ✨ What This Does

This tool lets you take any image (PNG, JPEG, WEBP…) and have it drawn automatically on your phone's Instagram story drawing canvas via Android Debug Bridge (ADB). It:

- **Quantises** your image to Instagram's exact 22-colour palette
- Lets you **place and resize** the drawing area over a live phone screenshot
- Shows a **layer-by-layer mapping dashboard** where you can remap colours, toggle fill vs outline modes, reorder layers, and pick a background colour
- Opens an **eraser editor** so you can clean up any unwanted pixels before anything is drawn
- **Draws** each colour layer in the correct order (lightest fills first, dark outlines on top) using ADB swipe commands

---

## 📋 Requirements

| Requirement | Details |
|---|---|
| **Python** | 3.8 or newer |
| **Android device** | Any Android phone with **USB Debugging** enabled |
| **ADB** | Android Platform Tools (see install below) |
| **Instagram** | Installed, story creation open with **Draw tool active** |
| **USB cable** | Phone connected to PC |

### Python packages

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
opencv-python>=4.5.0
numpy>=1.20.0
```

---

## 🔧 Setup

### Step 1 — Install Android Platform Tools

Download **Android Platform Tools** from Google and extract them:

- **Windows**: https://dl.google.com/android/repository/platform-tools-latest-windows.zip
- **macOS**: https://dl.google.com/android/repository/platform-tools-latest-darwin.zip
- **Linux**: https://dl.google.com/android/repository/platform-tools-latest-linux.zip

Place `adb` (or `adb.exe` on Windows) in the same folder as the Python scripts **or** add it to your system `PATH`.

### Step 2 — Enable USB Debugging on your phone

1. Go to **Settings → About Phone**
2. Tap **Build Number** 7 times to unlock Developer Options
3. Go to **Settings → Developer Options**
4. Enable **USB Debugging**
5. Connect phone via USB and accept the "Allow USB Debugging" prompt on the phone

### Step 3 — Clone this repo

```bash
git clone https://github.com/Sarty-085/instagram-auto-drawing-tool.git
cd instagram-auto-drawing-tool
pip install -r requirements.txt
```

### Step 4 — Calibrate your device *(first time only)*

Calibration detects where Instagram's palette bar sits on your specific screen and maps the brush-size slider positions. You only need to do this once per device.

```bash
python draw_interactive.py --calibrate path/to/any_image.png
```

The auto-calibration will:
1. Take a screenshot of your phone
2. Detect the colour palette row at the bottom of Instagram's draw view
3. Detect the brush-size slider on the left side
4. Save a `config.json` for future runs

> **Manual calibration**: If auto-calibration fails, run `python calibration.py` to go through a step-by-step guided process.

---

## 🚀 Usage

Open Instagram on your phone, create a new story, tap the **pen/draw icon**, and make sure you are on **palette page 1**.

Then run:

```bash
python draw_interactive.py path/to/your_image.png
```

### Full command options

```
python draw_interactive.py [IMAGE] [--calibrate] [--config CONFIG_PATH]

Arguments:
  IMAGE           Path to the image to draw (PNG, JPG, WEBP, etc.)
  --calibrate     Force re-calibration even if config.json exists
  --config PATH   Use a specific config file instead of the default
```

### Examples

```bash
# Draw a PNG image
python draw_interactive.py samples/sleeping.png

# Draw and re-calibrate first
python draw_interactive.py samples/bear.png --calibrate

# Draw using a custom config file
python draw_interactive.py my_art.png --config my_phone_config.json
```

---

## 🖥️ Interactive Workflow

Once launched, the tool walks you through 4 interactive windows:

---

### Window 1 — Overlay Placement

A window showing your phone screenshot with the image overlaid.

| Control | Action |
|---|---|
| **Drag image body** | Move the drawing area |
| **Drag red corner handle** | Resize (aspect-ratio locked) |
| **ENTER / SPACE** | Confirm placement |
| **ESC** | Cancel |

Position the image exactly where you want it drawn on the phone screen.

---

### Window 2 — Mapping Dashboard

A control panel where you customise how each colour layer is drawn.

| Control | Action |
|---|---|
| **Click a layer row** | Select it (preview highlights that layer) |
| **▲ / ▼ arrows** | Reorder layers |
| **Click colour swatch** | Remap to a different Instagram palette colour |
| **FILL / LINE toggle** | Switch between scanline fill and outline-only mode |
| **SKIP** | Exclude this colour entirely |
| **Background selector** | Set a solid background colour (or none) |
| **ENTER / SPACE** | Confirm and proceed |
| **ESC** | Cancel |

> **Tip**: Layers are pre-sorted lightest → darkest. This means light fills (white, cream) are drawn first and dark outlines (black) are drawn last — matching how cartoon art is traditionally layered.

---

### Window 3 — Eraser Editor

Before drawing starts you can clean up any unwanted pixels.

| Control | Action |
|---|---|
| **Left-click / drag** | Erase pixels from the active layer |
| **Right-click / drag** | Restore erased pixels |
| **`[` / `]`** | Shrink / grow eraser brush |
| **N / P** | Next / previous layer |
| **ENTER / SPACE** | Confirm edits and continue |
| **ESC** | Cancel — discards ALL edits and returns to dashboard |

Active layer shows at full brightness; other layers are dimmed so you can see context.

---

### Window 4 — Countdown & Drawing

Switch to your phone, make sure Instagram is open on the draw page, then wait for the countdown (default 3 seconds). The tool will:

1. Select the correct palette page and colour
2. Set the brush to its smallest size
3. Draw each layer using horizontal scanline sweeps (fill mode) or contour tracing (outline mode)
4. Automatically lift the pen between each shape to avoid unwanted connecting lines

---

## ⚙️ Configuration

After calibration a `config.json` is created in the project folder. You can edit it manually:

```jsonc
{
  "device": {
    "screen_width": 1080,         // Phone screen width in px
    "screen_height": 2408,        // Phone screen height in px
    "brush_slider_x": 42,         // X coordinate of the brush-size slider
    "palette_y": 2198,            // Y coordinate of the colour palette bar
    "safe_x_boundary": 120,       // Left margin to avoid the slider zone
    "palette_x_positions": [...]  // X positions of the 7 colour swatches
  },
  "brush_config": {
    "1": { "y": 1511, "width": 27 },   // Thinnest brush
    "2": { "y": 1350, "width": 42 },
    "3": { "y": 1176, "width": 69 },
    "4": { "y": 1022, "width": 96 },
    "5": { "y": 869,  "width": 114 }   // Thickest brush
  },
  "drawing": {
    "fill_step_size": 6,            // Pixels between scanlines (smaller = denser fill, slower)
    "contour_epsilon": 0.0012,      // Outline simplification factor
    "swipe_duration_long": 800,     // ms for long swipes (> 150 px)
    "swipe_duration_mid": 300,      // ms for medium swipes (50–150 px)
    "swipe_duration_short": 120,    // ms for short swipes (< 50 px)
    "inter_swipe_delay": 0.15,      // Seconds to wait between swipes (pen-lift time)
    "post_background_settle": 1.5,  // Seconds to wait after background fill
    "pre_draw_countdown": 3         // Seconds countdown before drawing starts
  },
  "palette_page_swipe": {
    "start_x": 850,
    "end_x": 450,
    "duration": 400,
    "settle_delay": 1.5
  }
}
```

> `config.json` is **gitignored** — it contains coordinates specific to your phone.

---

## 🎨 Instagram Colour Palette

The tool maps every pixel to the nearest of Instagram's 22 built-in drawing colours:

| # | Name | Swatch |
|---|---|---|
| 0 | White | ⬜ |
| 1–6 | Blue, Green, Yellow, Orange, Red, Pink | 🌈 |
| 7–13 | Purple, Instagram Red, Rose, Light Pink, Pale Orange, Peach, Gold Brown | 🌈 |
| 14 | Brown | 🟫 |
| 15 | Black | ⬛ |
| 16–21 | Dark Grey → Very Light Grey | 🩶 |

These span 4 palette pages in the Instagram drawing UI. The tool automatically swipes between pages when selecting colours.

---

## 🗂️ Project Structure

```
instagram-auto-drawing-tool/
│
├── draw_interactive.py      # 🚀 Main entry point — run this
│
├── drawing_engine.py        # Core ADB drawing logic (swipe paths, scanlines, fill)
├── gui.py                   # OpenCV interactive windows (overlay, dashboard, eraser)
├── calibration.py           # Auto + manual device calibration
├── calibrate_slider.py      # Brush-size slider calibration helper
├── measure_brush_sizes.py   # Utility to measure actual brush widths on screen
├── color_mapping.py         # Instagram palette definition + image quantisation
├── adb_utils.py             # ADB connection wrapper (tap, swipe, screenshot)
├── config.py                # Default config schema + load/save helpers
│
├── requirements.txt         # Python dependencies
├── instagram_palette.json   # Machine-readable palette definition
│
└── samples/                 # Example images to try
    ├── sleeping.png
    ├── bear.png
    ├── puppy.png
    └── ...
```

---

## 🛠️ Troubleshooting

### `adb` not found
Make sure `adb.exe` (Windows) or `adb` (Mac/Linux) is in the same folder as the scripts, or on your system `PATH`. Download it from [Android Platform Tools](https://developer.android.com/studio/releases/platform-tools).

### No device found
- Check USB cable is properly connected
- Ensure **USB Debugging** is on in Developer Options
- Accept the "Allow USB Debugging" prompt on the phone
- Try `adb devices` in a terminal to verify the device is listed

### Calibration fails / palette not detected
- Open Instagram, create a story, tap the **Draw** tool, and make sure the palette bar at the bottom is visible
- Make sure you're on **palette page 1** (scroll the palette to the left)
- Run `python draw_interactive.py --calibrate` to re-run calibration

### Drawing connects shapes with stray lines
The `inter_swipe_delay` in `config.json` may be too low for your device. Increase it:
```json
"inter_swipe_delay": 0.20
```

### Fill paints over wrong areas
- Use the **Eraser Editor** (Window 3) to remove unwanted pixels before drawing
- Switch problem layers from `FILL` to `LINE` mode in the dashboard

### Drawing is very slow
Increase `fill_step_size` in `config.json` (e.g. `10` or `12`) to use fewer scanlines. The fill will be less dense but much faster.


---

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Sarty-085/instagram-auto-drawing-tool&type=Date)](https://star-history.com/#Sarty-085/instagram-auto-drawing-tool&Date)

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Android Platform Tools](https://developer.android.com/studio/releases/platform-tools) (ADB) by Google
- [OpenCV](https://opencv.org/) for image processing and interactive GUI
- [NumPy](https://numpy.org/) for fast mask operations

