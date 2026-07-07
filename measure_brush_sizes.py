import cv2
import numpy as np
import subprocess
import time
import os

# --- HARDCODED BRUSH COORDINATES ---
BRUSH_X = 42
BRUSH_THIN_Y = 1511
BRUSH_MED_Y = 1176
BRUSH_THICK_Y = 869

# Calculate 5 equidistant Y coordinates on the slider
slider_y_coords = [
    1511,  # Size 1 (Thinnest)
    1350,  # Size 2
    1176,  # Size 3 (Medium)
    1022,  # Size 4
    869    # Size 5 (Thickest)
]

# ADB Command Utilities (Portable resolution)
def find_adb():
    """Dynamically resolves the path to adb.exe to make the app portable."""
    import sys
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
    local_adb = os.path.join(base_dir, "adb.exe")
    if os.path.exists(local_adb):
        return local_adb
        
    if os.path.exists("adb.exe"):
        return os.path.abspath("adb.exe")
        
    sarth_path = r"c:\Users\sarth\Downloads\platform-tools-latest-windows\platform-tools\adb.exe"
    if os.path.exists(sarth_path):
        return sarth_path
        
    from shutil import which
    system_adb = which("adb")
    if system_adb:
        return system_adb
        
    return "adb.exe"

ADB_PATH = find_adb()
print(f"Using ADB from path: {ADB_PATH}")

def tap_screen(x, y):
    cmd = f'"{ADB_PATH}" shell input tap {x} {y}'
    subprocess.run(cmd, shell=True)
    time.sleep(0.5)

def swipe_screen(x1, y1, x2, y2, duration=50):
    cmd = f'"{ADB_PATH}" shell input swipe {x1} {y1} {x2} {y2} {duration}'
    subprocess.run(cmd, shell=True)

def select_brush_size_raw(y_coord):
    """Drags the brush slider to a specific Y coordinate."""
    swipe_screen(BRUSH_X, y_coord, BRUSH_X + 15, y_coord, duration=200)
    time.sleep(0.5)

def get_phone_screenshot():
    print("Capturing phone screen via ADB...")
    cmd = [ADB_PATH, "exec-out", "screencap", "-p"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE)
    img_bytes = proc.stdout
    if not img_bytes or len(img_bytes) < 100:
        print("Error: Could not capture phone screen.")
        return None
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Click tracking globals
click_points = []

def mouse_callback(event, x, y, flags, param):
    global click_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale back to phone space
        scale = param['scale']
        phone_x = int(x / scale)
        phone_y = int(y / scale)
        click_points.append((phone_x, phone_y))
        print(f"Clicked pixel: X={phone_x}, Y={phone_y}")

def main():
    global click_points
    
    print("\n--- INSTRUCTIONS ---")
    print("1. Open Instagram Draw mode on your phone.")
    print("2. Choose a clean canvas (e.g. solid black or dark background).")
    print("3. Select WHITE color from the palette for maximum contrast.")
    print("4. Press ENTER when ready to draw the 5 calibration dots...")
    input()
    
    # Draw 5 dots on the phone screen
    # Placed at different Y values: 400, 750, 1100, 1450, 1800
    draw_y_coords = [400, 750, 1100, 1450, 1800]
    
    print("Drawing calibration dots on phone...")
    for idx, slider_y in enumerate(slider_y_coords):
        size_num = idx + 1
        draw_y = draw_y_coords[idx]
        print(f"   Dot {size_num}/5: Slider Y = {slider_y}, Tapping at Y = {draw_y}...")
        
        # Select brush size
        select_brush_size_raw(slider_y)
        
        # Tap the screen to draw a circular dot
        tap_screen(500, draw_y)
        time.sleep(0.5)
        
    print("Test dots drawn. Taking screenshot...")
    time.sleep(1.0)
    screenshot = get_phone_screenshot()
    if screenshot is None:
        return
        
    h_screen, w_screen = screenshot.shape[:2]
    
    # Scale display to fit laptop screen
    scale = 1.0
    max_h = 800
    if h_screen > max_h:
        scale = max_h / h_screen
        
    disp_w = int(w_screen * scale)
    disp_h = int(h_screen * scale)
    display_img = cv2.resize(screenshot, (disp_w, disp_h))
    
    window_name = "Brush Diameter Calibration Tool"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, param={'scale': scale})
    
    measured_widths = []
    
    print("\n--- MEASUREMENT STEPS ---")
    print("We will measure the 5 dots from top to bottom (Size 1 to Size 5).")
    
    for idx, draw_y in enumerate(draw_y_coords):
        size_num = idx + 1
        print(f"\n[Size {size_num}] Slider Y: {slider_y_coords[idx]}")
        print("-> Click on the LEFTmost edge of the dot in the window.")
        
        click_points = []
        # Wait for left click
        while len(click_points) < 1:
            canvas = display_img.copy()
            # Draw helper horizontal line to show where the dot Y is
            disp_y = int(draw_y * scale)
            cv2.line(canvas, (0, disp_y), (disp_w, disp_y), (0, 0, 255), 1)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                return
                
        pt_left = click_points[0]
        # Draw indicator for left click
        cv2.drawMarker(display_img, (int(pt_left[0] * scale), int(pt_left[1] * scale)), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
        
        print("-> Click on the RIGHTmost edge of the same dot.")
        # Wait for right click
        while len(click_points) < 2:
            canvas = display_img.copy()
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                return
                
        pt_right = click_points[1]
        cv2.drawMarker(display_img, (int(pt_right[0] * scale), int(pt_right[1] * scale)), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
        
        # Calculate width in pixels (diameter of the circle)
        width_pixels = abs(pt_right[0] - pt_left[0])
        measured_widths.append(width_pixels)
        print(f"Measured diameter for Size {size_num}: {width_pixels} pixels")
        
        # Draw bounding circle around it for verification
        cx = int((pt_left[0] + pt_right[0]) / 2 * scale)
        cy = int(draw_y * scale)
        r = int(width_pixels / 2 * scale)
        cv2.circle(display_img, (cx, cy), r, (255, 255, 0), 1)
        
    cv2.destroyAllWindows()
    
    print("\n==========================================")
    print("CALIBRATION COMPLETED!")
    print("==========================================")
    print("Copy and paste the following configuration dictionary into draw_interactive.py:")
    print("\n# --- CALIBRATED BRUSH CONFIGURATION ---")
    print("BRUSH_CONFIG = {")
    for i in range(5):
        size_num = i + 1
        y_val = slider_y_coords[i]
        width_val = measured_widths[i]
        print(f"    {size_num}: {{\"y\": {y_val}, \"width\": {width_val}}},  # Size {size_num}")
    print("}")
    print("==========================================\n")

if __name__ == "__main__":
    main()
