import cv2
import numpy as np
import subprocess
import os

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

# Mouse click callback
coords_clicked = []
img_display = None

def click_callback(event, x, y, flags, param):
    global img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        coords_clicked.append((x, y))
        print(f"Clicked Coordinates: X = {x}, Y = {y}")
        
        # Draw red circle and text label on display image
        cv2.circle(img_display, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(img_display, f"({x}, {y})", (x + 15, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Calibration Screen (ESC to Exit)", img_display)

def main():
    global img_display
    screenshot = get_phone_screenshot()
    if screenshot is None:
        return
        
    h, w = screenshot.shape[:2]
    print(f"Captured screen size: {w}x{h}")
    
    # Scale display image down so it fits laptop screen
    max_h = 800
    scale = 1.0
    if h > max_h:
        scale = max_h / h
        
    display_w = int(w * scale)
    display_h = int(h * scale)
    display_img = cv2.resize(screenshot, (display_w, display_h))
    img_display = display_img.copy()
    
    cv2.namedWindow("Calibration Screen (ESC to Exit)")
    cv2.setMouseCallback("Calibration Screen (ESC to Exit)", click_callback)
    
    print("\n--- INSTRUCTIONS ---")
    print("1. Find the brush size slider on the left edge of your phone screen.")
    print("2. Click on the BOTTOM of the slider (for THIN brush size).")
    print("3. Click on the MIDDLE of the slider (for MEDIUM brush size).")
    print("4. Click on the TOP of the slider (for THICK brush size).")
    print("5. Look at the coordinates printed in the terminal.")
    print("   Note: You must divide these clicked coordinates by the scale factor to get full phone coordinates!")
    print(f"   Scale Factor used: {scale:.4f}")
    print("   Formula: Phone_Coordinate = Clicked_Coordinate / Scale_Factor")
    print("\nPress ESC in the window to close when finished.")
    
    while True:
        cv2.imshow("Calibration Screen (ESC to Exit)", img_display)
        key = cv2.waitKey(15) & 0xFF
        if key == 27: # ESC
            break
            
    cv2.destroyAllWindows()
    
    if coords_clicked:
        print("\n--- FINAL CALIBRATED PHONE COORDINATES ---")
        for i, coord in enumerate(coords_clicked):
            phone_x = int(coord[0] / scale)
            phone_y = int(coord[1] / scale)
            print(f"Click {i+1}: Display({coord[0]}, {coord[1]}) -> Phone X={phone_x}, Y={phone_y}")

if __name__ == "__main__":
    main()
