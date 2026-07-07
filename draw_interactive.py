import cv2
import numpy as np
import subprocess
import time
import os

# --- DRAWING SPEED CONFIGURATION ---
# Swipe speed durations in ms based on line segment length
SWIPE_DURATION_LONG = 800   # Slow, stable strokes for long fills/backgrounds (increased for perfect registration)
SWIPE_DURATION_MID = 300    # Moderate speed for medium strokes
SWIPE_DURATION_SHORT = 120  # Fast speed for tiny detail segments

def get_swipe_duration(x1, y1, x2, y2):
    """Calculates swipe duration in ms dynamically based on distance to optimize drawing speed."""
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if dist > 150:
        return SWIPE_DURATION_LONG
    elif dist > 50:
        return SWIPE_DURATION_MID
    else:
        return SWIPE_DURATION_SHORT

# --- FILL CONFIGURATION ---
FILL_STEP_SIZE = 6  # Step size in pixels for solid coloring. Reverted to 6px for zero gaps.

# --- CONTOUR SMOOTHNESS CONFIGURATION ---
# Lower values break curves into more points/segments, making circles and curved outlines appear much smoother.
# Recommended values: 0.001 (ultra-smooth), 0.0015 (high-fidelity), 0.005 (simplified/blocky).
CONTOUR_EPSILON = 0.0012

# --- BRUSH SIZE SLIDER CONFIGURATION ---
# Run scratch/calibrate_slider.py to find these coordinates for your phone!
BRUSH_X = 42          # Horizontal position of the brush size slider

# --- CALIBRATED BRUSH CONFIGURATION ---
# Default settings for 5 brush sizes (Y coordinates and line width in pixels).
# Run scratch/measure_brush_sizes.py to get these exact values for your device!
BRUSH_CONFIG = {
    1: {"y": 1511, "width": 27},  # Size 1
    2: {"y": 1350, "width": 42},  # Size 2
    3: {"y": 1176, "width": 69},  # Size 3
    4: {"y": 1022, "width": 96},  # Size 4
    5: {"y": 869, "width": 114},  # Size 5
}

# --- INSTAGRAM PALETTE DEFINITION ---
# Mapping BGR colors to their Page and X coordinate
PALETTE_Y = 2200
COLORS_PALETTE = [
    # (B, G, R), Page, X coordinate, Name
    ((255, 255, 255), 1, 432, "white"),
    ((241, 151, 56), 1, 515, "blue"),
    ((79, 192, 112), 1, 597, "green"),
    ((65, 200, 254), 1, 680, "yellow"),
    ((51, 141, 252), 1, 763, "orange"),
    ((87, 73, 238), 1, 845, "red"),
    ((106, 7, 209), 1, 928, "pink"),
    
    ((186, 7, 162), 2, 432, "purple"),
    ((20, 0, 237), 2, 515, "instagram_red"),
    ((142, 133, 237), 2, 597, "rose"),
    ((212, 211, 255), 2, 680, "light_pink"),
    ((179, 219, 254), 2, 763, "pale_orange"),
    ((130, 196, 255), 2, 845, "peach"),
    ((70, 144, 210), 2, 928, "gold_brown"),
    
    ((58, 100, 153), 3, 432, "brown"),
    ((38, 38, 38), 3, 515, "black"),
    ((54, 54, 54), 3, 597, "dark_grey"),
    ((85, 85, 85), 3, 680, "grey"),
    ((115, 115, 115), 3, 763, "light_mid_grey"),
    ((153, 153, 153), 3, 845, "mid_grey"),
    ((178, 178, 178), 3, 928, "light_grey"),
    
    ((199, 199, 199), 4, 432, "very_light_grey")
]

# Track currently active color page (assume start on page 1)
current_page = 1

def find_adb():
    """Dynamically resolves the path to adb.exe to make the app portable."""
    import sys
    if getattr(sys, 'frozen', False):
        # Running as compiled PyInstaller executable
        base_dir = os.path.dirname(sys.executable)
    else:
        # Running as standard script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
    local_adb = os.path.join(base_dir, "adb.exe")
    if os.path.exists(local_adb):
        return local_adb
        
    # 2. Check current working directory
    if os.path.exists("adb.exe"):
        return os.path.abspath("adb.exe")
        
    # 3. Check hardcoded path for user sarth (backward compatibility)
    sarth_path = r"c:\Users\sarth\Downloads\platform-tools-latest-windows\platform-tools\adb.exe"
    if os.path.exists(sarth_path):
        return sarth_path
        
    # 4. Check system PATH
    from shutil import which
    system_adb = which("adb")
    if system_adb:
        return system_adb
        
    # Default fallback
    return "adb.exe"

ADB_PATH = find_adb()
print(f"Using ADB from path: {ADB_PATH}")

def get_phone_screenshot():
    print("Capturing phone screen via ADB...")
    cmd = [ADB_PATH, "exec-out", "screencap", "-p"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE)
    img_bytes = proc.stdout
    
    if not img_bytes or len(img_bytes) < 100:
        print("Error: Could not capture phone screen. Check connection and permissions.")
        return None
        
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def tap_screen(x, y):
    cmd = f'"{ADB_PATH}" shell input tap {x} {y}'
    subprocess.run(cmd, shell=True)
    time.sleep(0.5)

def swipe_screen(x1, y1, x2, y2, duration=50):
    cmd = f'"{ADB_PATH}" shell input swipe {x1} {y1} {x2} {y2} {duration}'
    subprocess.run(cmd, shell=True)

def go_to_page(target_page):
    global current_page
    if current_page == target_page:
        return
    
    print(f"   [Autonomous Page Swipe] Navigating from Page {current_page} to Page {target_page}...")
    while current_page < target_page:
        # Swipe left on the color bar (850 -> 450) to go to next page
        swipe_screen(850, PALETTE_Y, 450, PALETTE_Y, duration=400)
        time.sleep(1.5)
        current_page += 1
        
    while current_page > target_page:
        # Swipe right on the color bar (450 -> 850) to go to previous page
        swipe_screen(450, PALETTE_Y, 850, PALETTE_Y, duration=400)
        time.sleep(1.5)
        current_page -= 1

def select_color(palette_idx):
    color_bgr, page, x, name = COLORS_PALETTE[palette_idx]
    print(f"   Selecting color '{name}' (Page {page}, X={x})...")
    go_to_page(page)
    tap_screen(x, PALETTE_Y)
    time.sleep(0.5)

def select_brush_size(size_number):
    """Selects the brush size on the left edge slider using a short swipe drag based on config."""
    print(f"   Selecting brush size '{size_number}'...")
    if size_number not in BRUSH_CONFIG:
        print(f"   Warning: Invalid brush size {size_number}. Defaulting to size 1.")
        size_number = 1
        
    target_y = BRUSH_CONFIG[size_number]["y"]
    
    # Short swipe starting on slider and dragging slightly right to guarantee handle activation
    swipe_screen(BRUSH_X, target_y, BRUSH_X + 15, target_y, duration=200)
    time.sleep(0.8)

# --- TOUCH-DOWN SAFETY SYSTEM ---
def make_swipe_coordinates_safe(x1, y1, x2, y2, safe_x=120):
    """Ensures touch-down (x1, y1) never starts inside the slider zone (X < 120)."""
    if x1 < safe_x:
        if x2 >= safe_x:
            # Swipe ends in the safe zone: reverse direction to start at x2 (safe) and swipe left
            return x2, y2, x1, y1
        else:
            # Both start and end are in the slider zone. Shift start to safe_x boundary
            return safe_x, y1, x1, y2
    return x1, y1, x2, y2

def make_contour_safe(path, x_phone, safe_x=120):
    """Shifts the starting point of a closed outline loop so it touches down in the safe zone."""
    M = len(path)
    for i in range(M):
        px = int(path[i][0][0]) + x_phone
        if px >= safe_x:
            # Roll the array so index i becomes index 0
            return np.roll(path, -i, axis=0)
    return path  # Fallback if the entire contour is inside the slider zone

# --- COMPUTER VISION UTILITIES ---
def remove_background(img):
    """Detects and isolates background pixels using flood fill with fixed range thresholding."""
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    lodiff = (15, 15, 15)
    updiff = (15, 15, 15)
    
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    
    temp_img = img.copy()
    cv2.floodFill(temp_img, mask, (0, 0), (0, 0, 0), lodiff, updiff, flags=flags)
    cv2.floodFill(temp_img, mask, (w - 1, 0), (0, 0, 0), lodiff, updiff, flags=flags)
    cv2.floodFill(temp_img, mask, (0, h - 1), (0, 0, 0), lodiff, updiff, flags=flags)
    cv2.floodFill(temp_img, mask, (w - 1, h - 1), (0, 0, 0), lodiff, updiff, flags=flags)
    
    bg_mask = mask[1:-1, 1:-1] > 0
    return bg_mask

def prepare_source_image(img_path):
    """Loads image and adds an alpha channel with transparent background."""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
        
    if img.shape[2] == 4:
        print("Image loaded with existing alpha channel.")
        return img
        
    print("No alpha channel detected. Isolating background automatically...")
    bg_mask = remove_background(img)
    
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[bg_mask, 3] = 0 # Set alpha to 0 (transparent) for background
    
    return bgra

def quantize_image(img):
    """Maps every pixel in the image to the closest Instagram color."""
    h, w, c = img.shape
    pixels = img.reshape(-1, 3).astype(np.float32)
    palette_bgrs = np.array([item[0] for item in COLORS_PALETTE], dtype=np.float32)
    
    diff = pixels[:, np.newaxis, :] - palette_bgrs[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    closest_indices = np.argmin(dist_sq, axis=1)
    
    quantized_img = palette_bgrs[closest_indices].reshape(img.shape).astype(np.uint8)
    closest_indices_img = closest_indices.reshape((h, w))
    
    return quantized_img, closest_indices_img

def get_fill_paths_for_contour(contour, step_size, w_mask, h_mask):
    """Generates alternating horizontal scanlines to fill the interior of a shape."""
    temp_mask = np.zeros((h_mask, w_mask), dtype=np.uint8)
    cv2.drawContours(temp_mask, [contour], -1, 255, -1) # Draw solid filled shape
    
    paths = []
    direction = True # True: left-to-right, False: right-to-left
    
    x, y, w, h = cv2.boundingRect(contour)
    
    for scan_y in range(y, y + h, step_size):
        if scan_y >= h_mask:
            break
        row = temp_mask[scan_y, :]
        
        segments = []
        in_seg = False
        start_x = -1
        for scan_x in range(x, x + w):
            if scan_x >= w_mask:
                break
            if row[scan_x] > 0 and not in_seg:
                start_x = scan_x
                in_seg = True
            elif row[scan_x] == 0 and in_seg:
                segments.append((start_x, scan_x - 1))
                in_seg = False
        if in_seg:
            segments.append((start_x, min(x + w - 1, w_mask - 1)))
            
        if segments:
            if not direction:
                segments = [(seg[1], seg[0]) for seg in reversed(segments)]
            for seg in segments:
                paths.append(np.array([[[seg[0], scan_y]], [[seg[1], scan_y]]], dtype=np.int32))
            direction = not direction
            
    return paths

# --- GUI INTERACTIVE OVERLAY RENDERING ---
def overlay_image_bgra(background, foreground, x, y, w, h):
    """Blends transparent BGRA foreground onto BGR background at (x,y) with size (w,h)."""
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

# --- GUI INTERACTION GLOBALS (PLACEMENT PANEL) ---
dragging = False
resizing = False
start_mouse_x, start_mouse_y = 0, 0
start_overlay_x, start_overlay_y = 0, 0
start_overlay_w, start_overlay_h = 0, 0

# Active overlay position and size in display space
overlay_x, overlay_y = 50, 200
overlay_w, overlay_h = 200, 200

def mouse_callback(event, mx, my, flags, param):
    global dragging, resizing, start_mouse_x, start_mouse_y
    global start_overlay_x, start_overlay_y, start_overlay_w, start_overlay_h
    global overlay_x, overlay_y, overlay_w, overlay_h
    
    handle_size = 15
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if (overlay_x + overlay_w - handle_size <= mx <= overlay_x + overlay_w + handle_size and
            overlay_y + overlay_h - handle_size <= my <= overlay_y + overlay_h + handle_size):
            resizing = True
            start_mouse_x, start_mouse_y = mx, my
            start_overlay_w, start_overlay_h = overlay_w, overlay_h
        elif (overlay_x <= mx <= overlay_x + overlay_w and
              overlay_y <= my <= overlay_y + overlay_h):
            dragging = True
            start_mouse_x, start_mouse_y = mx, my
            start_overlay_x, start_overlay_y = overlay_x, overlay_y
            
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            dx = mx - start_mouse_x
            dy = my - start_mouse_y
            overlay_x = start_overlay_x + dx
            overlay_y = start_overlay_y + dy
        elif resizing:
            dx = mx - start_mouse_x
            new_w = max(50, start_overlay_w + dx)
            aspect_ratio = start_overlay_h / start_overlay_w
            new_h = int(new_w * aspect_ratio)
            overlay_w = new_w
            overlay_h = new_h
            
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        resizing = False

# --- GUI INTERACTION GLOBALS & CALLBACK (COLOR MAPPING PANEL) ---
selected_layer_idx = 0
present_color_indices = []
mapped_color_indices = list(range(22))
layer_modes = {}
scroll_offset = 0
bg_color_idx = -1  # -1 means no background color, otherwise maps to COLORS_PALETTE index

def mapping_mouse_callback(event, mx, my, flags, param):
    global selected_layer_idx, mapped_color_indices, present_color_indices
    global layer_modes, scroll_offset, bg_color_idx
    
    N = len(present_color_indices)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Click inside Middle Panel (Layers List)
        # x from 480 to 720
        if 480 <= mx <= 720:
            # Check click on BACKGROUND row (Y from 80 to 118)
            if 80 <= my <= 118:
                selected_layer_idx = "bg"
                print("Selected BACKGROUND layer")
                return
                
            # Otherwise check scrollable detected layers
            if N <= 9:
                # No scrolling
                for i in range(N):
                    y_min = 125 + i * 46
                    y_max = 125 + i * 46 + 38
                    if y_min <= my <= y_max:
                        handle_row_click(i, mx, y_min, y_max)
                        return
            else:
                # Scrolling active
                # Check click on Scroll Up button (Y from 125 to 150)
                if 125 <= my <= 150:
                    scroll_offset = max(0, scroll_offset - 1)
                    print(f"Scrolled UP (Offset: {scroll_offset})")
                    return
                # Check click on Scroll Down button (Y from 535 to 560)
                elif 535 <= my <= 560:
                    scroll_offset = min(N - 8, scroll_offset + 1)
                    print(f"Scrolled DOWN (Offset: {scroll_offset})")
                    return
                # Check click on visible rows
                for i in range(8):
                    y_min = 160 + i * 46
                    y_max = 160 + i * 46 + 38
                    if y_min <= my <= y_max:
                        handle_row_click(i + scroll_offset, mx, y_min, y_max)
                        return
                        
        # 2. Click inside Instagram Colors grid
        for idx, item in enumerate(COLORS_PALETTE):
            col = idx % 4
            row = idx // 4
            cx = 745 + col * 45 + 22
            cy = 85 + row * 50 + 22
            dist = np.sqrt((mx - cx)**2 + (my - cy)**2)
            if dist <= 18:
                if selected_layer_idx == "bg":
                    bg_color_idx = idx
                    print(f"Mapped BACKGROUND to color '{item[3]}'")
                    return
                elif selected_layer_idx is not None and selected_layer_idx < len(present_color_indices):
                    orig_idx = present_color_indices[selected_layer_idx]
                    mapped_color_indices[orig_idx] = idx
                    print(f"Mapped layer {selected_layer_idx + 1} to color '{item[3]}'")
                    return
                    
        # 3. Click on the Toggle Skip button (x from 480 to 720, y from 580 to 615)
        if 480 <= mx <= 720 and 580 <= my <= 615:
            if selected_layer_idx == "bg":
                if bg_color_idx == -1:
                    bg_color_idx = 0 # Default to white
                else:
                    bg_color_idx = -1 # Disable background
                print(f"Toggled BACKGROUND visibility to {bg_color_idx != -1}")
                return
            elif selected_layer_idx is not None and selected_layer_idx < len(present_color_indices):
                orig_idx = present_color_indices[selected_layer_idx]
                if mapped_color_indices[orig_idx] == -1:
                    mapped_color_indices[orig_idx] = orig_idx
                else:
                    mapped_color_indices[orig_idx] = -1
                print(f"Toggled skip/include for layer {selected_layer_idx + 1}")
                return

def handle_row_click(idx, mx, y_min, y_max):
    global selected_layer_idx, present_color_indices, layer_modes
    orig_idx = present_color_indices[idx]
    
    # Check click on Mode Button (565 <= mx <= 635)
    if 565 <= mx <= 635:
        layer_modes[orig_idx] = "outline" if layer_modes[orig_idx] == "fill" else "fill"
        print(f"Toggled Mode for layer {idx + 1} to {layer_modes[orig_idx]}")
        
    # Check click on Up reorder button (655 <= mx <= 685)
    elif 655 <= mx <= 685:
        if idx > 0:
            present_color_indices[idx], present_color_indices[idx - 1] = present_color_indices[idx - 1], present_color_indices[idx]
            selected_layer_idx = idx - 1
            print(f"Moved layer {idx + 1} UP")
            
    # Check click on Down reorder button (695 <= mx <= 725)
    elif 695 <= mx <= 725:
        if idx < len(present_color_indices) - 1:
            present_color_indices[idx], present_color_indices[idx + 1] = present_color_indices[idx + 1], present_color_indices[idx]
            selected_layer_idx = idx + 1
            print(f"Moved layer {idx + 1} DOWN")
            
    else:
        # Simple selection
        selected_layer_idx = idx
        print(f"Selected layer {idx + 1}")

def get_preview_image(closest_indices_img, fg_alpha, present_color_indices, mapped_indices, layer_modes, selected_idx, bg_color_idx):
    """Composites layers, drawing the background solid box first if enabled, and dimming non-selected colors."""
    h, w = closest_indices_img.shape
    preview = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 1. Draw Background Rectangle (if enabled)
    if bg_color_idx != -1:
        bg_bgr = np.array(COLORS_PALETTE[bg_color_idx][0], dtype=np.float32)
        if selected_idx != "bg":
            bg_bgr = bg_bgr * 0.2  # Dim background when not selected
        preview[:, :] = bg_bgr.astype(np.uint8)
        
    selected_orig_idx = present_color_indices[selected_idx] if (selected_idx is not None and selected_idx != "bg") else None
    
    # 2. Composite foreground layers in user reorder order
    for orig_idx in present_color_indices:
        mapped_idx = mapped_indices[orig_idx]
        if mapped_idx == -1: # Skip layer
            continue
            
        layer_mask = ((closest_indices_img == orig_idx) & (fg_alpha > 0))
        
        if layer_modes[orig_idx] == "outline":
            mask_uint8 = layer_mask.astype(np.uint8)
            dilated = cv2.dilate(mask_uint8, np.ones((3, 3), np.uint8))
            eroded = cv2.erode(mask_uint8, np.ones((3, 3), np.uint8))
            draw_mask = (dilated - eroded) > 0
        else:
            draw_mask = layer_mask
            
        # Get palette color BGR
        color_bgr = np.array(COLORS_PALETTE[mapped_idx][0], dtype=np.float32)
        
        # Highlight active layer: if a layer is selected, dim all other layers to 20% opacity
        if selected_idx != "bg" and selected_orig_idx is not None and orig_idx != selected_orig_idx:
            color_bgr = color_bgr * 0.2
        elif selected_idx == "bg":
            # Dim all foreground elements if background is selected
            color_bgr = color_bgr * 0.2
            
        preview[draw_mask] = color_bgr.astype(np.uint8)
        
    return preview

def draw_row(canvas, idx, orig_idx, y_min, y_max, is_selected, mapped_color_indices, layer_modes):
    """Draws a single layer row with Mode selector and reorder buttons (brush size selector removed)."""
    bg_color = (75, 75, 75) if is_selected else (45, 43, 42)
    border_color = (0, 255, 255) if is_selected else (80, 80, 80) # Cyan highlight
    
    cv2.rectangle(canvas, (485, y_min), (725, y_max), bg_color, -1)
    cv2.rectangle(canvas, (485, y_min), (725, y_max), border_color, 2 if is_selected else 1)
    
    # Original swatch
    orig_bgr = COLORS_PALETTE[orig_idx][0]
    cv2.rectangle(canvas, (492, y_min + 7), (512, y_max - 7), orig_bgr, -1)
    cv2.rectangle(canvas, (492, y_min + 7), (512, y_max - 7), (255, 255, 255), 1)
    
    # Arrow
    cv2.putText(canvas, ">", (518, y_min + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
    
    # Mapped swatch or SKIP
    mapped_idx = mapped_color_indices[orig_idx]
    if mapped_idx == -1:
        cv2.putText(canvas, "SKIP", (530, y_min + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 255), 1, cv2.LINE_AA)
    else:
        mapped_bgr = COLORS_PALETTE[mapped_idx][0]
        cv2.rectangle(canvas, (530, y_min + 7), (550, y_max - 7), mapped_bgr, -1)
        cv2.rectangle(canvas, (530, y_min + 7), (550, y_max - 7), (255, 255, 255), 1)
        
    # Mode Pill (FILL / LINE) - Expanded width
    mode_text = "FILL" if layer_modes[orig_idx] == "fill" else "LINE"
    mode_btn_bg = (0, 90, 0) if layer_modes[orig_idx] == "fill" else (100, 40, 0)
    cv2.rectangle(canvas, (565, y_min + 6), (635, y_max - 6), mode_btn_bg, -1)
    cv2.rectangle(canvas, (565, y_min + 6), (635, y_max - 6), (200, 200, 200), 1)
    cv2.putText(canvas, mode_text, (585, y_min + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Reorder buttons side by side - Expanded width
    # Up Button
    cv2.rectangle(canvas, (655, y_min + 6), (685, y_max - 6), (60, 60, 60), -1)
    cv2.rectangle(canvas, (655, y_min + 6), (685, y_max - 6), (200, 200, 200), 1)
    cv2.putText(canvas, "^", (667, y_min + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Down Button
    cv2.rectangle(canvas, (695, y_min + 6), (725, y_max - 6), (60, 60, 60), -1)
    cv2.rectangle(canvas, (695, y_min + 6), (725, y_max - 6), (200, 200, 200), 1)
    cv2.putText(canvas, "v", (707, y_min + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

def draw_mapping_panel(preview_img, present_color_indices, selected_layer_idx, mapped_color_indices, layer_modes, scroll_offset, bg_color_idx):
    """Draws the 960x640 Premium Dashboard Panel with scrolling, highlights, and BACKGROUND layer."""
    canvas = np.zeros((640, 960, 3), dtype=np.uint8)
    canvas[:] = (34, 32, 31) # Dark theme background
    
    # 1. Title Header
    cv2.putText(canvas, "INSTAGRAM DOODLE BOT - DASHBOARD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 2. Separation Grid Lines
    cv2.line(canvas, (475, 70), (475, 570), (70, 70, 70), 1)
    cv2.line(canvas, (735, 70), (735, 570), (70, 70, 70), 1)
    cv2.line(canvas, (20, 570), (940, 570), (70, 70, 70), 1)
    
    # Sub-headings
    cv2.putText(canvas, "Live Preview", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Layers Config", (485, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Instagram Palette", (745, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    
    # 3. Live Preview Frame
    preview_resized = cv2.resize(preview_img, (440, 440))
    canvas[85:525, 20:460] = preview_resized
    cv2.rectangle(canvas, (20, 85), (460, 525), (100, 100, 100), 2)
    
    # 4. Draw BACKGROUND Row (always fixed at Y = 80 to 118)
    is_bg_selected = (selected_layer_idx == "bg")
    bg_row_bg = (85, 85, 85) if is_bg_selected else (45, 43, 42)
    bg_border_color = (0, 255, 255) if is_bg_selected else (80, 80, 80)
    
    cv2.rectangle(canvas, (485, 80), (725, 118), bg_row_bg, -1)
    cv2.rectangle(canvas, (485, 80), (725, 118), bg_border_color, 2 if is_bg_selected else 1)
    
    if bg_color_idx == -1:
        # Crossed out background swatch (disabled)
        cv2.rectangle(canvas, (492, 87), (512, 111), (30, 30, 30), -1)
        cv2.rectangle(canvas, (492, 87), (512, 111), (100, 100, 100), 1)
        cv2.line(canvas, (492, 87), (512, 111), (0, 0, 255), 2)
    else:
        mapped_bg_bgr = COLORS_PALETTE[bg_color_idx][0]
        cv2.rectangle(canvas, (492, 87), (512, 111), mapped_bg_bgr, -1)
        cv2.rectangle(canvas, (492, 87), (512, 111), (255, 255, 255), 1)
        
    cv2.putText(canvas, ">", (518, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(canvas, "BACKGROUND", (530, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Static mode selector for BACKGROUND (FILL is always active)
    cv2.rectangle(canvas, (565, 86), (635, 112), (0, 90, 0), -1)
    cv2.rectangle(canvas, (565, 86), (635, 112), (200, 200, 200), 1)
    cv2.putText(canvas, "FILL", (585, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Separation line between BACKGROUND and scrollable doodle layers
    cv2.line(canvas, (485, 122), (725, 122), (90, 90, 90), 1)
    
    # 5. Middle Column: Detected Layers List (scrolling starts at Y = 125)
    N = len(present_color_indices)
    if N <= 9:
        # Fits completely, draw starting at Y = 125
        for i in range(N):
            y_min = 125 + i * 46
            y_max = 125 + i * 46 + 38
            draw_row(canvas, i, present_color_indices[i], y_min, y_max, selected_layer_idx == i, mapped_color_indices, layer_modes)
    else:
        # Draw Scroll Up Button (Y from 125 to 150)
        is_up_disabled = (scroll_offset == 0)
        up_bg = (50, 50, 50) if is_up_disabled else (70, 70, 220)
        cv2.rectangle(canvas, (485, 125), (725, 150), up_bg, -1)
        cv2.rectangle(canvas, (485, 125), (725, 150), (120, 120, 120), 1)
        cv2.putText(canvas, "▲ Scroll Up", (565, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw 8 visible rows
        for i in range(8):
            idx = i + scroll_offset
            if idx >= N:
                break
            y_min = 160 + i * 46
            y_max = 160 + i * 46 + 38
            draw_row(canvas, idx, present_color_indices[idx], y_min, y_max, selected_layer_idx == idx, mapped_color_indices, layer_modes)
            
        # Draw Scroll Down Button (Y from 535 to 560)
        is_down_disabled = (scroll_offset >= N - 8)
        down_bg = (50, 50, 50) if is_down_disabled else (70, 70, 220)
        cv2.rectangle(canvas, (485, 535), (725, 560), down_bg, -1)
        cv2.rectangle(canvas, (485, 535), (725, 560), (120, 120, 120), 1)
        cv2.putText(canvas, "▼ Scroll Down", (560, 552), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
    # Draw Toggle Skip Button at bottom
    cv2.rectangle(canvas, (485, 580), (725, 615), (40, 40, 180), -1)
    cv2.rectangle(canvas, (485, 580), (725, 615), (255, 255, 255), 1)
    cv2.putText(canvas, "TOGGLE SKIP / INCLUDE", (515, 602), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 6. Right Column: Instagram Palette Grid
    for idx, item in enumerate(COLORS_PALETTE):
        col = idx % 4
        row = idx // 4
        cx = 745 + col * 45 + 22
        cy = 85 + row * 50 + 22
        
        cv2.circle(canvas, (cx, cy), 18, item[0], -1)
        cv2.circle(canvas, (cx, cy), 18, (255, 255, 255), 1)
        
        # Glowing selection indicator
        target_matched = False
        if selected_layer_idx == "bg":
            target_matched = (bg_color_idx == idx)
        elif selected_layer_idx is not None and selected_layer_idx < len(present_color_indices):
            orig_idx = present_color_indices[selected_layer_idx]
            target_matched = (mapped_color_indices[orig_idx] == idx)
            
        if target_matched:
            cv2.circle(canvas, (cx, cy), 6, (0, 255, 0), -1)
            cv2.circle(canvas, (cx, cy), 9, (0, 255, 0), 1)
            cv2.circle(canvas, (cx, cy), 20, (0, 255, 0), 2) # Outer ring glow
            
    # 7. Bottom Footer Instructions
    cv2.putText(canvas, "ENTER: Start drawing on phone | ESC: Cancel drawing", (20, 595), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Click rows to edit. Reorder elements so background fills are drawn first.", (20, 615), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    
    return canvas

# --- MAIN EXECUTION ---
def main():
    global current_page
    global overlay_x, overlay_y, overlay_w, overlay_h
    global selected_layer_idx, present_color_indices, mapped_color_indices
    global layer_modes, scroll_offset, bg_color_idx
    
    # 1. Prompt for image path
    image_path = input("Enter path to the image to draw (default: sleeping.png): ").strip()
    if not image_path:
        image_path = "sleeping.png"
        
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return
        
    source_bgra = prepare_source_image(image_path)
    if source_bgra is None:
        print("Error: Could not load or prepare image.")
        return
        
    src_h, src_w = source_bgra.shape[:2]
    
    # 2. Capture phone screen screenshot
    screenshot = get_phone_screenshot()
    if screenshot is None:
        return
        
    h_screen, w_screen = screenshot.shape[:2]
    
    # 3. Setup downscaled display image to fit laptop screens (max height 800px)
    scale_factor = 1.0
    max_display_h = 800
    if h_screen > max_display_h:
        scale_factor = max_display_h / h_screen
    
    display_w = int(w_screen * scale_factor)
    display_h = int(h_screen * scale_factor)
    display_screenshot = cv2.resize(screenshot, (display_w, display_h))
    
    # Initialize overlay size and position in display space
    overlay_w = int(display_w * 0.5)
    overlay_h = int(overlay_w * (src_h / src_w))
    overlay_x = int((display_w - overlay_w) / 2)
    overlay_y = int((display_h - overlay_h) / 2)
    
    # 4. Open Interactive Placement Window
    window_name = "Overlay Editor Panel (ENTER to Confirm, ESC to Cancel)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\n--- ACTION REQUIRED ---")
    print("A placement panel window has opened.")
    print("- CLICK & DRAG the image to move it.")
    print("- DRAG the bottom-right red corner square to resize it (aspect ratio locked).")
    print("- Press ENTER or SPACE in the window to confirm drawing position.")
    print("- Press ESC to cancel.")
    
    while True:
        canvas = display_screenshot.copy()
        
        # Overlay the transparent image
        canvas = overlay_image_bgra(canvas, source_bgra, overlay_x, overlay_y, overlay_w, overlay_h)
        
        # Draw bounding border (Green)
        cv2.rectangle(canvas, (overlay_x, overlay_y), (overlay_x + overlay_w, overlay_y + overlay_h), (0, 255, 0), 2)
        # Draw resize handle (Red square)
        cv2.rectangle(canvas, (overlay_x + overlay_w - 6, overlay_y + overlay_h - 6),
                      (overlay_x + overlay_w + 6, overlay_y + overlay_h + 6), (0, 0, 255), -1)
        
        # Display guidelines text
        cv2.putText(canvas, "Drag body to Move | Drag Red corner to Resize", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, "Drag body to Move | Drag Red corner to Resize", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Press ENTER/SPACE to Confirm | Press ESC to Cancel", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, "Press ENTER/SPACE to Confirm | Press ESC to Cancel", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(window_name, canvas)
        
        key = cv2.waitKey(15) & 0xFF
        if key == 13 or key == 32: # Enter or Space
            break
        elif key == 27 or key == ord('c') or key == ord('q'): # ESC, c, q
            print("Interactive placement cancelled by user.")
            cv2.destroyAllWindows()
            return
            
    cv2.destroyAllWindows()
    
    # 5. Translate coordinates back to full phone screen space
    x_phone = int(overlay_x / scale_factor)
    y_phone = int(overlay_y / scale_factor)
    w_phone = int(overlay_w / scale_factor)
    h_phone = int(overlay_h / scale_factor)
    
    print(f"\nFinal phone coordinates: X={x_phone}, Y={y_phone}, W={w_phone}, H={h_phone}")
    
    # 6. Prepare foreground for quantization and tracing
    resized_fg = cv2.resize(source_bgra, (w_phone, h_phone))
    
    fg_bgr = resized_fg[:, :, :3]
    fg_bgr = cv2.medianBlur(fg_bgr, 3)
    fg_alpha = resized_fg[:, :, 3] # Binary alpha mask
    
    # 7. Quantize BGR colors to the Instagram palette
    print("Quantizing BGR colors...")
    quantized_bgr, closest_indices_img = quantize_image(fg_bgr)
    
    # 8. Filter noise layers and identify active foreground layers
    unique_indices, counts = np.unique(closest_indices_img[fg_alpha > 0], return_counts=True)
    
    # Exclude color layers making up less than 1.5% of the foreground (anti-aliasing noise)
    total_fg_pixels = np.sum(fg_alpha > 0)
    MIN_LAYER_PIXELS = max(100, int(total_fg_pixels * 0.015))
    MIN_CONTOUR_AREA = 15.0 # Exclude tiny noise contours
    
    # Sort unique layers by count descending
    sorted_pairs = sorted(zip(unique_indices, counts), key=lambda x: -x[1])
    
    # Populate present color layers, filtering out layers below the pixel threshold
    present_color_indices = [idx for idx, count in sorted_pairs if count >= MIN_LAYER_PIXELS]
    
    if not present_color_indices:
        print("Error: No visible layers detected in foreground.")
        return
        
    # Reset mapped indices and background color index
    mapped_color_indices = list(range(22))
    selected_layer_idx = 0
    scroll_offset = 0
    bg_color_idx = -1  # None/disabled by default
    
    # Initialize draw modes: large layers (>3000 pixels) -> fill, others -> outline
    layer_modes = {}
    for orig_idx, count in sorted_pairs:
        layer_modes[orig_idx] = "fill" if count > 3000 else "outline"
            
    # 9. Open Color Mapping GUI Panel
    mapping_window = "Color Mapping Panel (ENTER to Confirm, ESC to Cancel)"
    cv2.namedWindow(mapping_window)
    cv2.setMouseCallback(mapping_window, mapping_mouse_callback)
    
    print("\n--- ACTION REQUIRED ---")
    print("The Drawing Dashboard window has opened.")
    print("- Click on the 'BACKGROUND' row at the top to customize background coloring.")
    print("- Click on any row in 'Layers Config' (Middle) to select a layer.")
    print("  Note: The selected layer stands out in full brightness in the Live Preview, while other layers are dimmed.")
    print("- Click '▲ Scroll Up' / '▼ Scroll Down' if you have more than 10 layers to view hidden ones.")
    print("- Click '▲' (Up) / 'v' (Down) buttons on the right of any row to reorder drawing layers.")
    print("- Click 'FILL'/'LINE' to toggle solid coloring or outline tracing.")
    print("- Click any circle in 'Instagram Palette' (Right) to remap the color.")
    print("- Click 'TOGGLE SKIP' to exclude/include a layer.")
    print("- Press ENTER or SPACE in the window to confirm and start drawing on your phone.")
    print("- Press ESC to cancel.")
    
    while True:
        # Generate the composite preview image highlighting the selected layer (and background if enabled)
        preview_img = get_preview_image(closest_indices_img, fg_alpha, present_color_indices, mapped_color_indices, layer_modes, selected_layer_idx, bg_color_idx)
        
        # Render the dashboard panel
        panel_canvas = draw_mapping_panel(preview_img, present_color_indices, selected_layer_idx, mapped_color_indices, layer_modes, scroll_offset, bg_color_idx)
        
        cv2.imshow(mapping_window, panel_canvas)
        
        key = cv2.waitKey(15) & 0xFF
        if key == 13 or key == 32: # Enter or Space
            break
        elif key == 27 or key == ord('c') or key == ord('q'): # ESC, c, q
            print("Color mapping panel cancelled by user.")
            cv2.destroyAllWindows()
            return
            
    cv2.destroyAllWindows()
    
    # 10. Pre-generate Background Box Paths if enabled
    bg_paths = []
    if bg_color_idx != -1:
        direction = True
        step_size = 18  # Scanline step size for size 5 (thickest) brush
        for scan_y in range(0, h_phone, step_size):
            start_x = 0
            end_x = w_phone - 1
            if not direction:
                start_x, end_x = end_x, start_x
            bg_paths.append(np.array([[[start_x, scan_y]], [[end_x, scan_y]]], dtype=np.int32))
            direction = not direction

    # 11. Tracing and Drawing Execution on phone (layer-by-layer in user order)
    print("\nMake sure Instagram Draw is open on your phone and scroll to Page 1 of colors!")
    print("Starting drawing process in 3 seconds...")
    time.sleep(3)
    
    # A. Draw Solid Background Rectangle FIRST (if enabled)
    if bg_color_idx != -1:
        print(f"\nDrawing solid BACKGROUND ({COLORS_PALETTE[bg_color_idx][3]})...")
        select_color(bg_color_idx)
        select_brush_size(5)  # Use Size 5 (thickest) for background
        
        for shape_idx, path in enumerate(bg_paths):
            start_x = int(path[0][0][0]) + x_phone
            start_y = int(path[0][0][1]) + y_phone
            next_x = int(path[1][0][0]) + x_phone
            next_y = int(path[1][0][1]) + y_phone
            
            # Enforce touch-down safety: redirect start coordinate away from left slider
            safe_x1, safe_y1, safe_x2, safe_y2 = make_swipe_coordinates_safe(start_x, start_y, next_x, next_y)
            swipe_screen(safe_x1, safe_y1, safe_x2, safe_y2, duration=get_swipe_duration(safe_x1, safe_y1, safe_x2, safe_y2))
            time.sleep(0.02)
        print("Finished drawing solid background. Pausing 1.5 seconds to let device settle...")
        time.sleep(1.5)
            
    # B. Draw Foreground Doodle Layers
    for orig_idx in present_color_indices:
        mapped_idx = mapped_color_indices[orig_idx]
        if mapped_idx == -1: # Skip layer
            print(f"Skipping layer '{COLORS_PALETTE[orig_idx][3]}' per user request.")
            continue
            
        mask = ((closest_indices_img == orig_idx) & (fg_alpha > 0)).astype(np.uint8) * 255
        mode = layer_modes[orig_idx]
        
        # Find contours of shapes in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if there are valid contours
        has_shapes = False
        for contour in contours:
            if cv2.contourArea(contour) >= MIN_CONTOUR_AREA:
                has_shapes = True
                break
                
        if not has_shapes:
            continue
            
        print(f"\nDrawing layer '{COLORS_PALETTE[mapped_idx][3]}' (Mode: {mode.upper()})...")
        select_color(mapped_idx)
        select_brush_size(1)  # Always use thinnest Size 1 brush for doodle elements
        
        if mode == "fill":
            # 1. SOLID FILL PASS ONLY (No outlines for fill mode)
            print("   Drawing solid fill scanlines...")
            fill_paths = []
            for contour in contours:
                if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                    continue
                paths = get_fill_paths_for_contour(contour, FILL_STEP_SIZE, w_phone, h_phone)
                fill_paths.extend(paths)
                
            for shape_idx, path in enumerate(fill_paths):
                start_x = int(path[0][0][0]) + x_phone
                start_y = int(path[0][0][1]) + y_phone
                next_x = int(path[1][0][0]) + x_phone
                next_y = int(path[1][0][1]) + y_phone
                
                safe_x1, safe_y1, safe_x2, safe_y2 = make_swipe_coordinates_safe(start_x, start_y, next_x, next_y)
                swipe_screen(safe_x1, safe_y1, safe_x2, safe_y2, duration=get_swipe_duration(safe_x1, safe_y1, safe_x2, safe_y2))
                time.sleep(0.02)
        else:
            # 2. OUTLINE PASS ONLY (For LINE mode only)
            print("   Tracing outlines...")
            outline_paths = []
            for contour in contours:
                if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                    continue
                epsilon = CONTOUR_EPSILON * cv2.arcLength(contour, True)
                approx_path = cv2.approxPolyDP(contour, epsilon, False)
                if len(approx_path) >= 2:
                    approx_path = make_contour_safe(approx_path, x_phone)
                    contour_points = [approx_path[pt_idx][0] for pt_idx in range(len(approx_path))]
                    contour_points.append(approx_path[0][0])
                    outline_paths.append(contour_points)
                    
            for shape_idx, path in enumerate(outline_paths):
                print(f"      Outline loop {shape_idx + 1}/{len(outline_paths)}...")
                start_x = int(path[0][0]) + x_phone
                start_y = int(path[0][1]) + y_phone
                
                for i in range(1, len(path)):
                    next_x = int(path[i][0]) + x_phone
                    next_y = int(path[i][1]) + y_phone
                    
                    safe_x1, safe_y1, safe_x2, safe_y2 = make_swipe_coordinates_safe(start_x, start_y, next_x, next_y)
                    swipe_screen(safe_x1, safe_y1, safe_x2, safe_y2, duration=get_swipe_duration(safe_x1, safe_y1, safe_x2, safe_y2))
                    start_x = next_x
                    start_y = next_y
                    time.sleep(0.02)
                
    print("\nDrawing completed successfully!")

if __name__ == "__main__":
    main()
