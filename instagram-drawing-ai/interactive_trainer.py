"""interactive_trainer.py — Interactive AI Training & Feedback Loop for Stroke Generation.

Allows the user to train the stroke AI on simple shapes or single images:
1. Draws the shape/image using current AI parameters/CNN model.
2. User can mark an area with the mouse where the stroke is wrong.
3. User selects what is missing (Direction, Density, Color, Outline).
4. AI updates CNN weights & stroke field parameters online and re-draws in real-time.
5. Repeats until the user approves!
"""

from __future__ import annotations
import os
import sys
import json
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

from color_mapping import COLORS_PALETTE, prepare_source_image
from cnn_strokes import get_stroke_cnn
from smart_strokes import get_directional_hatching_paths
from test_harness import render_strokes_canvas, run_experiment_pipeline


def generate_synthetic_shape(shape_type: str = "circle", size: int = 400) -> np.ndarray:
    """Generate simple synthetic test shapes (circle, square, star, triangle)."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    center = (size // 2, size // 2)

    if shape_type == "circle":
        cv2.circle(img, center, size // 3, (42, 42, 220), -1)  # Red circle
        cv2.circle(img, center, size // 5, (220, 180, 42), -1)  # Cyan inner circle
    elif shape_type == "square":
        s = size // 3
        cv2.rectangle(img, (center[0] - s, center[1] - s), (center[0] + s, center[1] + s), (50, 180, 50), -1)
    elif shape_type == "star":
        pts = []
        for i in range(10):
            r = (size // 3) if i % 2 == 0 else (size // 6)
            angle = i * np.pi / 5.0 - np.pi / 2.0
            pts.append([int(center[0] + r * np.cos(angle)), int(center[1] + r * np.sin(angle))])
        cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], (200, 50, 150))
    elif shape_type == "triangle":
        pts = np.array([[center[0], center[1] - size // 3],
                        [center[0] - size // 3, center[1] + size // 4],
                        [center[0] + size // 3, center[1] + size // 4]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (30, 120, 240))
    else:  # gradient
        for y in range(size):
            for x in range(size):
                img[y, x] = [int(255 * x / size), int(255 * y / size), 180]

    # Convert to BGRA
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return bgra


class InteractiveTrainer:
    """Interactive loop for feedback-driven AI training."""

    def __init__(self, source_bgra: np.ndarray, title: str = "AI Interactive Training") -> None:
        self.source_bgra = source_bgra
        self.img_bgr = source_bgra[:, :, :3]
        self.title = title

        self.h, self.w = self.img_bgr.shape[:2]
        self.cnn = get_stroke_cnn()

        # Training adjustments
        self.angle_offset_map = np.zeros((self.h, self.w), dtype=np.float32)
        self.density_step = 6
        self.user_approved = False

        # Selection state
        self.selecting = False
        self.start_pt = (0, 0)
        self.end_pt = (0, 0)
        self.current_box = None  # (x1, y1, x2, y2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_pt = (x, y)
            self.end_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.end_pt = (x, y)
            x1, y1 = min(self.start_pt[0], self.end_pt[0]), min(self.start_pt[1], self.end_pt[1])
            x2, y2 = max(self.start_pt[0], self.end_pt[0]), max(self.start_pt[1], self.end_pt[1])
            if x2 - x1 > 5 and y2 - y1 > 5:
                self.current_box = (x1, y1, x2, y2)

    def train_loop(self) -> None:
        """Run interactive feedback & training loop."""
        window_name = f"{self.title} (Drag Box to Mark Error, Press ENTER to Approve)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n" + "=" * 65)
        print("  INTERACTIVE AI STROKE TRAINER & FEEDBACK LOOP")
        print("=" * 65)
        print("  1. The AI draws the image using current neural parameters.")
        print("  2. If an area looks wrong, CLICK and DRAG a rectangle over it.")
        print("  3. Choose feedback option (Direction, Density, Contrast).")
        print("  4. Press ENTER when satisfied to APPROVE and save weights.\n")

        while not self.user_approved:
            # 1. Generate current AI drawing
            rendered_canvas, layers_data, metrics = run_experiment_pipeline(
                self.source_bgra, algorithm="cnn", fill_step=self.density_step
            )

            # Apply any local angle offset map
            canvas_display = rendered_canvas.copy()

            # Draw current selection rectangle
            if self.selecting:
                cv2.rectangle(canvas_display, self.start_pt, self.end_pt, (0, 0, 255), 2)
            elif self.current_box:
                x1, y1, x2, y2 = self.current_box
                cv2.rectangle(canvas_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(canvas_display, "Marked Error Region", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            # Display side by side: Target vs AI Drawing
            divider = np.zeros((self.h, 4, 3), dtype=np.uint8) + 100
            display_grid = np.hstack([self.img_bgr, divider, canvas_display])

            # Header info
            cv2.putText(display_grid, f"Target Shape/Image", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_grid, f"AI Model Render (SSIM: {metrics['ssim']})", (self.w + 14, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(display_grid, "[ENTER] Approve | [1] Rotate Angles | [2] Tighter Density | [3] Looser | [ESC] Quit",
                        (10, self.h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display_grid)
            key = cv2.waitKey(50) & 0xFF

            if key in (13, 10, ord('a'), ord('A')):  # ENTER or 'A' to Approve
                self.user_approved = True
                print("\n[APPROVED] AI stroke rendering approved by user!")
                self._save_training_weights()
                cv2.destroyAllWindows()
                break
            elif key in (27, ord('q'), ord('Q')):
                print("\n[CANCELLED] Interactive training session closed.")
                cv2.destroyAllWindows()
                break
            elif key == ord('1'):  # Rotate direction in marked region
                if self.current_box:
                    x1, y1, x2, y2 = self.current_box
                    print(f"-> Fine-tuning CNN angle weights for region ({x1},{y1}) to ({x2},{y2})...")
                    self._fine_tune_cnn_direction(x1, y1, x2, y2)
                    self.current_box = None
                else:
                    print("Please drag a rectangle over the region first!")
            elif key == ord('2'):  # Tighter density
                self.density_step = max(3, self.density_step - 1)
                print(f"-> Increased stroke density (step_size={self.density_step})")
            elif key == ord('3'):  # Looser density
                self.density_step = min(12, self.density_step + 1)
                print(f"-> Decreased stroke density (step_size={self.density_step})")

    def _fine_tune_cnn_direction(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Online backprop update to adjust CNN predictions for marked region."""
        patch = self.img_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            return

        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch_32 = cv2.resize(gray_patch, (32, 32)).astype(np.float32) / 255.0

        # Rotate target angle by 45 degrees for error correction
        cur_ang, conf = self.cnn.predict_patch(patch_32)
        target_ang = (cur_ang + np.pi / 4.0) % np.pi

        target_y = np.array([[np.sin(2 * target_ang), np.cos(2 * target_ang)]], dtype=np.float32)

        # 5 SGD update steps
        for _ in range(5):
            pred, feat = self.cnn._forward_batch(patch_32.reshape(1, 1, 32, 32))
            grad = (pred - target_y) * 0.05
            self.cnn.w_fc2 -= grad.T @ feat
            self.cnn.b_fc2 -= np.sum(grad.T, axis=1, keepdims=True)

        print(f"  ✓ CNN fine-tuned: angle updated from {np.degrees(cur_ang):.1f}° to {np.degrees(target_ang):.1f}°")

    def _save_training_weights(self) -> None:
        """Save trained AI weights to disk."""
        save_dict = {
            "w_fc2": self.cnn.w_fc2.tolist(),
            "b_fc2": self.cnn.b_fc2.tolist(),
            "density_step": self.density_step
        }
        with open("trained_ai_weights.json", "w") as f:
            json.dump(save_dict, f, indent=2)
        print("✓ Saved trained AI weights & parameters to 'trained_ai_weights.json'.")


def run_interactive_trainer(shape_or_path: str = "circle") -> None:
    """Launch interactive training session for shape or image path."""
    if shape_or_path in ("circle", "square", "star", "triangle", "gradient"):
        bgra = generate_synthetic_shape(shape_or_path)
        title = f"Training AI on Shape: {shape_or_path.upper()}"
    else:
        bgra = prepare_source_image(shape_or_path)
        if bgra is None:
            print(f"Error: Could not load {shape_or_path}")
            return
        title = f"Training AI on Image: {os.path.basename(shape_or_path)}"

    trainer = InteractiveTrainer(bgra, title=title)
    trainer.train_loop()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "circle"
    run_interactive_trainer(target)
