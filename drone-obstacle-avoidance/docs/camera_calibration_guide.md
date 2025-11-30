# Camera Calibration Guide

## Overview

Calibrating the drone’s camera with pre-captured chessboard images is fully supported. Capturing images ahead of time offers reproducibility, higher quality, and better control over lighting and framing. This guide preserves every original instruction while restructuring the content for clarity and providing a complete English translation.

---

## Benefits of Capturing Images in Advance

1. **Reusability** – the curated image set can be processed repeatedly with different parameters.
2. **Quality Control** – frames can be reviewed, re-shot, and validated before calibration.
3. **Higher Resolution** – smartphone sensors usually exceed laptop webcams in resolution.
4. **Archiving** – the dataset becomes a standard reference for future experiments.

---

## Capture Requirements

### Checkerboard Specification

- **Pattern**: 6 columns × 5 rows of inner corners (7 × 6 squares).

### Core Shot List (15–20 Images Recommended)

| #     | View / Placement        | Distance      | Frame Coverage | Notes                                           |
| ----- | ----------------------- | ------------- | -------------- | ----------------------------------------------- |
| 1–2   | Frontal, board parallel | 0.3 m / 0.6 m | 60–80%         | Keep chessboard parallel to the camera.         |
| 3–4   | Yaw left 30°            | 0.4 m / 0.5 m | 50–70%         | Rotate around vertical axis to the left.        |
| 5–6   | Yaw right 30°           | 0.4 m / 0.5 m | 50–70%         | Rotate around vertical axis to the right.       |
| 7–8   | Pitch up 30°            | 0.4 m / 0.5 m | 50–70%         | Board’s top edge tilts away from the camera.    |
| 9–10  | Pitch down 30°          | 0.4 m / 0.5 m | 50–70%         | Board’s bottom edge tilts away from the camera. |
| 11    | Top-left corner         | 0.5 m         | 40–50%         | Board positioned in the image’s top-left area.  |
| 12    | Top-right corner        | 0.5 m         | 40–50%         | Board positioned in the image’s top-right area. |
| 13    | Bottom-left corner      | 0.5 m         | 40–50%         | Board positioned in the bottom-left area.       |
| 14    | Bottom-right corner     | 0.5 m         | 40–50%         | Board positioned in the bottom-right area.      |
| 15    | Center, 45° rotation    | 0.5 m         | 50–60%         | Board remains parallel but rotated in-plane.    |
| 16–20 | Free combinations       | 0.3–0.7 m     | 40–80%         | Mix diverse angles, distances, and coverage.    |

### Distance Reference

- Use a tape measure from lens to board plane.
- **0.3 m** ≈ full arm’s length.
- **0.5 m** ≈ half-meter ruler.
- **0.7 m** ≈ one adult stride.

---

## Critical Notes

### 0. Distance and Coverage

- ✅ Ensure all four board corners remain visible with 10–20% border margin.
- ✅ Coverage can vary between 40–80%; diversity is beneficial.
- ❌ Do not crop the board or push it flush against frame edges.

### 1. Lighting

- ✅ Prefer even indoor lighting or diffused daylight.
- ❌ Avoid harsh sunlight, reflections, or strong shadows.
- ✅ Tilt slightly if needed to remove glare.

### 2. Focus and Sharpness

- ✅ Tap to focus on the board; keep the phone steady (tripod recommended).
- ❌ Reject blurry or motion-distorted frames.

### 3. Board Placement

- ✅ Keep the entire chessboard inside the frame with consistent margins.
- ❌ Prevent clipping, even partially.

### 4. Camera Settings (Phone)

- ✅ Disable HDR, beauty filters, and color filters.
- ✅ Use the primary rear camera at maximum resolution (≥1920×1080).
- ❌ Avoid ultra-wide or telephoto lenses.

### 5. Checkerboard Condition

- ✅ Mount on flat backing to prevent bending.
- ✅ Maintain high contrast between black and white squares.
- ❌ Avoid wrinkles, stains, or damage.

---

## Recommended Directory and Naming Scheme

Use descriptive filenames to track pose and distance combinations.

```bash
# Suggested directory layout
calibration_images/
├── front_near.jpg
├── front_far.jpg
├── left_tilt_30_near.jpg
├── left_tilt_30_far.jpg
├── right_tilt_30_near.jpg
├── right_tilt_30_far.jpg
├── up_tilt_30_near.jpg
├── up_tilt_30_far.jpg
├── down_tilt_30_near.jpg
├── down_tilt_30_far.jpg
├── corner_top_left.jpg
├── corner_top_right.jpg
├── corner_bottom_left.jpg
├── corner_bottom_right.jpg
├── center_rotate_45.jpg
├── combo_1.jpg
├── combo_2.jpg
├── combo_3.jpg
├── combo_4.jpg
└── combo_5.jpg
```
