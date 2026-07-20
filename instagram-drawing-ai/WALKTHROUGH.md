# Instagram Auto-Drawing AI Experiment — Walkthrough

> **Goal:** Make the Instagram auto-drawing tool produce more artistic, less mechanical-looking drawings by improving color matching, stroke patterns, and fill algorithms — all as a local experiment separate from the main GitHub repo.

---

## Current State (Baseline)

The existing tool works like this:

1. **Load image** → detect/remove background via flood-fill from corners
2. **Quantize colors** → map every pixel to the nearest of Instagram's 22 colors using Euclidean distance in RGB space
3. **Generate layers** → one binary mask per palette color
4. **Draw** → horizontal scanline sweeps (zig-zag left→right, right→left) for fill mode, or contour tracing for outline mode
5. **Send to phone** → ADB swipes execute the strokes on Instagram's canvas

### Why it looks "flat"

| Problem | Cause |
|---------|-------|
| Uniform horizontal stripes | `get_fill_paths_from_mask()` uses fixed horizontal scanlines regardless of image content |
| Color banding | Only 22 colors; no dithering to simulate intermediate tones |
| Poor color matches | RGB Euclidean distance doesn't match human perception (e.g., dark green vs dark blue) |
| Mechanical precision | Strokes are perfectly straight and evenly spaced — no artist-like variation |

---

## Phase 1: Perceptual Color Quantization

### What we're changing

Replace the RGB Euclidean distance in `quantize_image()` with **CIELAB ΔE 2000** — the industry-standard perceptual color difference metric.

### Why it matters

RGB distance treats red→green and blue→yellow as mathematically similar, but human eyes don't. CIELAB is designed to match human vision. Two colors that look very different might be close in RGB space, and vice versa.

### Implementation steps

1. Convert the 22-color Instagram palette from BGR → CIELAB
2. Convert input image pixels from BGR → CIELAB
3. Compute ΔE 2000 between each pixel and each palette color
4. Pick the palette color with smallest ΔE

### Bonus: Smart fallback for missing colors

If an input color is very far from ALL 22 palette colors (ΔE > threshold), instead of forcing the closest bad match:
- Flag it as "unmappable"
- In a later phase, use **dithering** to approximate it with two nearby palette colors

### Files to modify

- `color_mapping.py` — `quantize_image()` function
- Add `skimage.color` dependency for BGR→LAB conversion

---

## Phase 2: Directional Hatching Fill

### What we're changing

Replace horizontal scanline fill with **gradient-aware directional hatching**. Strokes follow the image's local structure instead of being uniformly horizontal.

### The algorithm

For each color layer mask:

1. **Compute distance transform** → how far each pixel is from the mask boundary
2. **Compute image gradients** (Sobel operators) on the original image within the mask region
3. **Stroke orientation** = perpendicular to gradient direction (i.e., follow edges, not cross them)
4. **Stroke spacing** = adaptive — tighter where detail is high (high gradient magnitude), looser in flat regions
5. **Generate strokes** as polylines following the orientation field, clipped to the mask

### Visual intuition

| Region type | Old behavior | New behavior |
|-------------|-------------|--------------|
| Curved edge (e.g., face outline) | Horizontal lines cutting across | Strokes following the curve |
| Flat area (e.g., shirt) | Uniform horizontal stripes | Looser, maybe still horizontal |
| Detail area (e.g., eyes) | Same spacing as everywhere | Tighter, more strokes |
| Diagonal structure | Horizontal lines crossing it | Strokes aligned with the diagonal |

### Why this looks better

Traditional artists use hatching — parallel strokes that follow form. A face drawn with strokes following the jawline and cheekbone looks organic. Horizontal stripes look like a printer output.

### Files to modify

- `drawing_engine.py` — new `get_directional_paths_from_mask()` function
- Keep old `get_fill_paths_from_mask()` for comparison

---

## Phase 3: Dithering for In-Between Colors

### What we're changing

When a color doesn't exist in the 22-color palette, instead of snapping to the closest match, use **ordered dithering** or **error diffusion** to simulate the missing color by alternating two nearby palette colors in a pattern.

### Example

Input color: teal (between green and blue)
- Old: snap to green OR snap to blue → wrong either way
- New: checkerboard pattern of green and blue dots → eye blends them into teal

### Algorithm: Bayer ordered dithering

1. For each pixel, compute its "ideal" color (before quantization)
2. Compare to two nearest palette colors: `color_a` (below) and `color_b` (above)
3. Use a Bayer threshold matrix: if pixel value > threshold, use `color_b`, else `color_a`
4. The spatial pattern makes the eye average the two colors perceptually

### Why ordered dithering over error diffusion

| Method | Pros | Cons |
|--------|------|------|
| Ordered dithering (Bayer) | Fast, deterministic, works per-pixel, no bleed between strokes | Slight regular pattern visible |
| Error diffusion (Floyd-Steinberg) | Smoother, less pattern | Requires raster order, harder to adapt to our stroke-based drawing |

Since we're generating strokes, not rendering pixels, ordered dithering is easier to adapt: we can pre-compute which pixels get which color, then generate strokes per sub-mask.

### Implementation

1. After perceptual quantization, identify "dither candidates": pixels where ΔE to nearest palette color > threshold
2. For those pixels, assign to `color_a` or `color_b` based on Bayer matrix position
3. This creates 2 sub-masks where there was 1 — both drawn with the same color layer but as separate passes

### Files to modify

- `color_mapping.py` — add `dither_quantize_image()` variant
- `drawing_engine.py` — handle dithered sub-layers

---

## Phase 4: Test Harness (No Phone Needed)

### What we're building

A visual comparison tool that simulates the drawing output without needing ADB or a phone.

### How it works

1. **Stroke renderer**: Take the generated stroke paths and render them as lines onto a blank canvas
   - Variable width per brush size
   - Slight random jitter to simulate human imperfection (optional)
   - Opacity blending for overlapping strokes

2. **Side-by-side comparison**:
   - Left: original image
   - Center: old algorithm output (horizontal scanlines)
   - Right: new algorithm output (directional hatching + dithering)

3. **Metrics**:
   - SSIM (structural similarity) vs original
   - Perceptual color distance (mean ΔE)
   - Stroke count (efficiency)

### Why this matters

Without a test harness, every iteration requires:
- Phone connected
- Instagram open
- ADB working
- 5+ minutes per test draw

With a harness: instant feedback, iterate on algorithms quickly.

### Files to create

- `test_harness.py` — stroke renderer + comparison window
- `metrics.py` — SSIM, ΔE, stroke count

---

## Phase 5 (Future): Tiny CNN for Stroke Direction

If Phases 1-3 aren't enough, add a tiny neural network:

- **Input**: 32×32 patch of the original image
- **Output**: stroke direction (angle 0-180°) + confidence
- **Architecture**: 3-layer CNN, ~50K parameters
- **Training data**: synthetic — render random shapes with known optimal hatching, or trace hatching from artistic line art datasets
- **Inference**: runs locally in <10ms per patch

This replaces the hand-crafted gradient-based direction estimation with learned patterns.

---

## File Structure (Experiment)

```
instagram-drawing-ai/
├── WALKTHROUGH.md              # This file
├── README.md                   # Copied from main repo
│
├── Core (copied + modified):
│   ├── draw_interactive.py     # Main entry point
│   ├── drawing_engine.py         # + directional hatching
│   ├── color_mapping.py         # + perceptual quantization + dithering
│   ├── gui.py                   # + test harness mode
│   ├── calibration.py
│   ├── adb_utils.py
│   ├── config.py
│   └── instagram_palette.json
│
├── New files:
│   ├── smart_strokes.py         # Directional hatching engine
│   ├── perceptual_color.py      # CIELAB quantization
│   ├── dithering.py             # Bayer / error diffusion
│   ├── test_harness.py          # Visual comparison tool
│   └── metrics.py               # Quality metrics
│
└── samples/                    # Test images
```

---

## Implementation Order

1. ✅ **Create workspace** — copy files, verify baseline works
2. ✅ **Perceptual color** — CIELAB & CIEDE2000 quantization (`perceptual_color.py`)
3. ✅ **Test harness** — offline renderer & visual metrics comparison (`test_harness.py`, `metrics.py`)
4. ✅ **Directional hatching** — gradient-aware streamline hatching (`smart_strokes.py`)
5. ✅ **Dithering** — Bayer 4x4 & 8x8 ordered dithering (`dithering.py`)
6. ✅ **Phase 5 Tiny CNN** — 3-layer CNN neural network stroke direction predictor (`cnn_strokes.py`)

---

## Success Criteria

| Metric | Baseline | Target |
|--------|----------|--------|
| Mean ΔE (perceptual color error) | ~15 | <8 |
| SSIM vs original | ~0.65 | >0.75 |
| "Mechanical" appearance | Obvious scanlines | Looks hand-hatched |
| Stroke count | N | ≤ 1.5× N (don't be too slow) |

---

## Notes

- **Keep main repo clean** — this experiment never gets committed to GitHub
- **No GPU required** — all algorithms run on CPU with OpenCV + NumPy
- **Optional dependency**: `scikit-image` for CIELAB conversion (can also pre-compute LUT)
- **Fallback**: if a technique fails, old horizontal scanlines are always available as `mode='legacy'`
