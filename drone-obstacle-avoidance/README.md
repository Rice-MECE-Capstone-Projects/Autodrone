# Drone Obstacle Avoidance System

Jetson-based autonomous navigation pipeline that combines Ultralytics YOLO11s object detection with multiple monocular depth estimators (Depth Anything V2, MiDaS, ZoeDepth, Apple Depth Pro) and streams avoidance commands to a Pixhawk flight controller via MAVLink. This README is the single source of truth for project structure, setup, calibration, execution, troubleshooting, and auxiliary tooling.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Repository Layout](#repository-layout)
3. [Prerequisites](#prerequisites)
4. [Environment Setup](#environment-setup)
5. [Quickstart Workflow](#quickstart-workflow)
6. [Shell Script Reference](#shell-script-reference)
7. [Python Pipeline Reference](#python-pipeline-reference)
8. [Calibration & Diagnostic Utilities](#calibration--diagnostic-utilities)
9. [Camera Calibration Workflow](#camera-calibration-workflow)
10. [Depth Model Validation](#depth-model-validation)
11. [Pixhawk / MAVLink Integration](#pixhawk--mavlink-integration)
12. [Storage & Docker Management](#storage--docker-management)
13. [Outputs & Artifacts](#outputs--artifacts)
14. [Troubleshooting](#troubleshooting)
15. [Reference Documents](#reference-documents)
16. [Authors](#authors)

---

## System Overview

- **Hardware**: NVIDIA Jetson (Orin/AGX/Nano), Pixhawk autopilot, USB RGB camera or Intel RealSense.
- **Detection**: Ultralytics YOLO11s (TensorRT-capable).
- **Depth Engines**: Depth Anything V2, MiDaS Small, ZoeDepth, Apple Depth Pro, MiDaS Pro metric converters.
- **Control**: MAVLink position/velocity targets in BODY_NED or LOCAL_NED frames.
- **Calibration**: Offline or live chessboard capture, automatic undistortion during runtime.
- **Utilities**: Storage/Docker management, GPU monitors, chessboard diagnostics, depth benchmarks.

---

## Repository Layout

```
drone-obstacle-avoidance/
├── checkpoints/                    # All downloaded model weights (YOLO, depth, etc.)
├── docs/                           # Markdown guides (camera calibration, Pixhawk wiring)
├── depth_pro/                      # Apple Depth Pro source (pip editable)
├── utils/                          # Calibration, diagnostics, Docker helpers
├── avoidance.py                    # YOLO11s + Depth Anything V2 baseline
├── avoidance_midas.py              # YOLO11s + MiDaS Small (relative depth)
├── avoidance_midas_pro.py          # MiDaS + metric converters (scene/size/geometry)
├── avoidance_midas_pro_v2.py       # MiDaS + advanced metric schemes (affine/SfM/etc.)
├── avoidance_zoedepth.py           # YOLO11s + ZoeDepth metric depth
├── avoidance_final.py              # Final demo: MiDaS + affine calibration + MAVLink
├── init.sh | run.sh | clean.sh     # Setup, execution stub, cleanup
├── config.yaml                     # L4T/JetPack compatibility metadata
└── camera_calib.*                  # Generated calibration artifacts (npz, txt, jpg)
```

---

## Prerequisites

| Component                 | Details                                                  |
| ------------------------- | -------------------------------------------------------- |
| Jetson (Orin/AGX/Nano)    | Ubuntu 20.04/22.04, JetPack ≥ 5.1, CUDA + cuDNN.         |
| Python 3.8+               | `pip3` with sudo access.                                 |
| NVIDIA Drivers            | Ensure `torch.cuda.is_available()` returns true.         |
| Optional: Intel RealSense | Install librealsense SDK for hardware depth.             |
| Optional: ROS 2 Humble    | Only for MAVROS testing (`ros2 run mavros mavros_node`). |
| GCS Software              | Mission Planner or QGroundControl for Pixhawk tuning.    |
| Storage                   | Recommend ≥ 1 TB NVMe for checkpoints & Docker images.   |

---

## Environment Setup

```bash
# 1. Clone meta-repo (if not already)
cd ~/capstone/jetson-examples
pip install -e .

# 2. Enter project directory
cd ~/capstone/jetson-examples/reComputer/scripts/drone-obstacle-avoidance

# 3. Install models and dependencies
bash init.sh
```

`init.sh` installs `numpy<2`, `ultralytics`, `opencv-python`, `pymavlink`, `torch`, `torchvision`, downloads `checkpoints/yolo11s.pt`, caches Depth Anything V2 under `~/.cache/depth_anything_v2/`.

---

## Quickstart Workflow

```bash
# (Optional) clean previous caches
bash clean.sh

# 1. Calibrate camera (live capture)
python3 utils/calibrate_camera.py

# 2. Sanity-check depth model
python3 utils/test_midas.py   # or test_depth_v2.py / test_depth_pro.py

# 3. Run baseline avoidance (RGB camera, no RealSense)
python3 avoidance.py --no-realsense --camera 0 --save-video

# 4. Run MiDaS Pro V2 with affine metric conversion and MAVLink UDP stream
python3 avoidance_midas_pro_v2.py \
  --method affine_invariant \
  --camera 0 \
  --no-realsense \
  --mavlink udp:192.168.1.100:14550 \
  --save-video
```

Use `run.sh` as a shortcut wrapper: `bash run.sh --no-realsense --camera 0`.

---

## Shell Script Reference

| Script                      | Purpose & Notes                                                                 | Command                                 |
| --------------------------- | ------------------------------------------------------------------------------- | --------------------------------------- |
| `init.sh`                   | Installs Python deps, downloads YOLO11s + Depth Anything V2, clones DA-V2 repo. | `bash init.sh`                          |
| `run.sh`                    | Convenience launcher for `avoidance.py` (passes all CLI args).                  | `bash run.sh --no-realsense --camera 0` |
| `clean.sh`                  | Deletes YOLO cache, Depth Anything V2 cache, output MP4/JPG files.              | `bash clean.sh`                         |
| `utils/setup_depth_pro.sh`  | Installs Apple Depth Pro (editable) and downloads `checkpoints/depth_pro.pt`.   | `cd utils && bash setup_depth_pro.sh`   |
| `utils/setup_docker_hdd.sh` | Moves Docker data-root to `/mnt/hdd/docker`, restarts Docker, optional cleanup. | `bash utils/setup_docker_hdd.sh`        |
| `utils/check_before_run.sh` | Verifies `/mnt/hdd` mount, Docker root dir, disk space, Docker service status.  | `bash utils/check_before_run.sh`        |
| `utils/fix_depth_v2.sh`     | Patches reComputer container launcher for Depth Anything V2.                    | `bash utils/fix_depth_v2.sh`            |

---

## Python Pipeline Reference

| File                        | Depth Backend                                                                                       | Key Features                                                                                                        | Typical Command                                                                                    |
| --------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `avoidance.py`              | Depth Anything V2 (ViT-S) or RealSense depth                                                        | Fastest baseline, optional depth disable, MAVLink preview.                                                          | `python3 avoidance.py --no-realsense --camera 0 --save-video`                                      |
| `avoidance_midas.py`        | MiDaS Small relative depth                                                                          | Lightweight, GPU-friendly, optional RealSense fallback.                                                             | `python3 avoidance_midas.py --camera 0 --no-realsense --quiet`                                     |
| `avoidance_midas_pro.py`    | MiDaS + metric converters (`scene_calib`, `size_priors`, `geometry`)                                | Interactive calibration (`C` key), loads `camera_calib.npz`, undistorts frames.                                     | `python3 avoidance_midas_pro.py --method size_priors --camera 0 --save-video`                      |
| `avoidance_midas_pro_v2.py` | MiDaS + advanced converters (`affine_invariant`, `least_squares`, `monocular_sfm`, `neural_refine`) | Runtime focal-length scaling, multiple metric strategies, UDP/serial MAVLink.                                       | `python3 avoidance_midas_pro_v2.py --method affine_invariant --camera 0 --mavlink /dev/ttyACM0`    |
| `avoidance_zoedepth.py`     | ZoeDepth (metric)                                                                                   | Real-scale distances, good for precise avoidance when GPU headroom available.                                       | `python3 avoidance_zoedepth.py --camera 0 --save-video`                                            |
| `avoidance_final.py`        | MiDaS + affine calibration + BODY_NED MAVLink                                                       | Final demo pipeline: on-start calibration ROI (`known_distance`), BODY-frame velocity output, minimal dependencies. | `python3 avoidance_final.py --camera 0 --no-realsense --mavlink /dev/ttyACM0 --known-distance 1.0` |

Common CLI flags (all pipelines):

- `--camera <id>`: USB camera index.
- `--no-realsense`: Force RGB pipeline even if RealSense is connected.
- `--no-depth-model`: YOLO-only debug.
- `--save-video`, `--output <file>`: Write MP4.
- `--mavlink <endpoint>`: `/dev/ttyACM0`, `/dev/ttyUSB0`, `udp:IP:14550`, or `dummy`.
- `--no-calibration`: Skip loading `camera_calib.npz` (for MiDaS Pro variants).
- `--quiet`: Reduce console spam.

---

## Calibration & Diagnostic Utilities

| Utility                              | Function                                                                                                                                                                              | Command                                                                                 |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `utils/calibrate_camera.py`          | Live capture with chessboard (6×5 inner corners). Press `SPACE` to record, `Q` to finish. Generates `camera_calib.npz`, `camera_calib.txt`, `calibration_test.jpg`, annotated frames. | `python3 utils/calibrate_camera.py`                                                     |
| `utils/calibrate_camera_new.py`      | Offline calibration from folder (smartphone shots). Supports `--square-size` and `--output`.                                                                                          | `python3 utils/calibrate_camera_new.py --images calibration_images --square-size 0.029` |
| `utils/check_all_images.py`          | Batch verify chessboard detection (6×5 & swapped 5×6).                                                                                                                                | `python3 utils/check_all_images.py`                                                     |
| `utils/diagnose_image.py`            | Analyze a single photo, test multiple board sizes, produce annotated result.                                                                                                          | `python3 utils/diagnose_image.py path/to/image.jpg`                                     |
| `utils/quick_verify.py`              | Simple live detection (6×5). Handy for checking printed board quality.                                                                                                                | `python3 utils/quick_verify.py`                                                         |
| `utils/test_chessboard_detection.py` | Interactive GUI to cycle board sizes (keys 1–9), save debug frames, view detection methods.                                                                                           | `python3 utils/test_chessboard_detection.py`                                            |
| `utils/test_depth_v2.py`             | Depth Anything V2 benchmark: threaded capture, FPS/CPU/GPU overlay, screenshot with `S`.                                                                                              | `python3 utils/test_depth_v2.py`                                                        |
| `utils/test_midas.py`                | MiDaS Small dashboard (actual resolution scaling, GPU monitor).                                                                                                                       | `python3 utils/test_midas.py`                                                           |
| `utils/test_depth_pro.py`            | Apple Depth Pro low-memory runner (uses `checkpoints/depth_pro.pt`).                                                                                                                  | `python3 utils/test_depth_pro.py`                                                       |

---

## Camera Calibration Workflow

1. **Capture Strategy** (see `docs/camera_calibration_guide.md`):

   - Chessboard: 6 columns × 5 rows of inner corners (7×6 squares).
   - Collect 15–20 shots covering frontal, yaw/pitch variations, corners, rotations.
   - Maintain 10–20% border margin; keep board flat, avoid glare.

2. **Live Calibration**

   ```bash
   python3 utils/calibrate_camera.py
   # Press SPACE for each valid pose, Q to finish (≥15 images recommended).
   ```

3. **Offline Calibration**

   ```bash
   python3 utils/calibrate_camera_new.py \
     --images calibration_images \
     --square-size 0.029 \
     --output camera_calib.npz
   ```

4. **Outputs**

   - `camera_calib.npz`: intrinsic matrix, distortion coefficients, image size, reprojection error.
   - `camera_calib.txt`: human-readable report (fx, fy, cx, cy, k1–k3, p1–p2).
   - `calibration_test.jpg`: side-by-side undistortion preview.
   - `calibration_images/annotated/`: annotated detections for audit.

5. **Runtime Usage**
   - MiDaS Pro scripts auto-load `camera_calib.npz`, undistort frames, and scale `fx/fy` to actual capture resolution. Console logs show the scaled parameters.

---

## Depth Model Validation

| Model             | Command                                                                    | Notes                                                                            |
| ----------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Depth Anything V2 | `python3 utils/test_depth_v2.py`                                           | Requires DA-V2 repo cloned under `~/.cache/depth_anything_v2/Depth-Anything-V2`. |
| MiDaS Small       | `python3 utils/test_midas.py`                                              | Reports actual resolution; warm-up ensures stable FPS.                           |
| Apple Depth Pro   | `bash utils/setup_depth_pro.sh` (once) → `python3 utils/test_depth_pro.py` | Uses `checkpoints/depth_pro.pt`; warns if VRAM < 3 GB.                           |

Each tester displays FPS, inference latency, CPU/GPU usage, and stores screenshots via key `S`. Use them before flight tests to confirm GPU health.

---

## Pixhawk / MAVLink Integration

1. **Serial Exclusivity**

   - `/dev/ttyACM0` can be opened by **either** MAVROS **or** the avoidance script, not both simultaneously.

2. **MAVROS Baseline**

   ```bash
   ros2 run mavros mavros_node --ros-args \
     -p fcu_url:=serial:///dev/ttyACM0:115200 \
     -p fcu_protocol:=v2.0 \
     -p target_system_id:=1 \
     -p target_component_id:=1
   ```

   Check `/mavros/state`, `/mavros/rc/out`, `/mavros/vfr_hud`, `/mavros/statustext/recv`.

3. **Mission Planner Visibility**

   - Configure Pixhawk `SERIALx_PROTOCOL=2`, `SERIALx_BAUD=115`, enable GCS forwarding via `SERIALx_OPTIONS` or `BRD_OPTIONS`.
   - If telemetry is still missing, stream UDP directly from Jetson:
     ```bash
     python3 avoidance_midas_pro_v2.py \
       --method affine_invariant \
       --camera 0 \
       --no-realsense \
       --mavlink udp:192.168.1.100:14550
     ```

4. **Mode & Arming Services**

   ```bash
   ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \
     "{base_mode: 0, custom_mode: 'GUIDED'}"
   ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \
     "{base_mode: 0, custom_mode: 'GUIDED_NOGPS'}"
   ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"
   ```

5. **Telemetry Stream Rates**

   ```bash
   ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval \
     "{message_id: 36, message_rate: 10.0}"   # SERVO_OUTPUT_RAW
   ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval \
     "{message_id: 74, message_rate: 5.0}"    # VFR_HUD
   ros2 service call /mavros/set_message_interval mavros_msgs/srv/MessageInterval \
     "{message_id: 253, message_rate: 2.0}"   # STATUSTEXT (event-driven)
   ```

6. **Final Demo (`avoidance_final.py`)**
   ```bash
   python3 avoidance_final.py \
     --camera 0 \
     --no-realsense \
     --mavlink /dev/ttyACM0 \
     --known-distance 1.5 \
     --save-video
   ```
   - On startup, highlight center ROI for calibration; press `c` when object at known distance is centered.
   - BODY_NED velocities: slows or sidesteps when obstacles enter front ROI.
   - Ideal for tethered bench validation (props removed).

---

## Storage & Docker Management

1. **Mount High-Capacity Disk**

   ```bash
   sudo mkdir -p /mnt/hdd
   sudo mount /dev/nvme0n1p1 /mnt/hdd
   ```

2. **Relocate Docker**

   ```bash
   bash utils/setup_docker_hdd.sh
   # Enter /mnt/hdd when prompted, optionally copy old /var/lib/docker
   ```

3. **Pre-Run Checklist**

   ```bash
   bash utils/check_before_run.sh
   ```

   Ensures `/mnt/hdd` is mounted, Docker root is `/mnt/hdd/docker`, disk space is sufficient, Docker daemon active.

4. **Cleaning Up**
   - Use `bash clean.sh` for model caches.
   - Remove old Docker data only after confirming new root: `sudo rm -rf /var/lib/docker`.

---

## Outputs & Artifacts

| Artifact                                                       | Description                                                    |
| -------------------------------------------------------------- | -------------------------------------------------------------- |
| `drone_avoidance_output.mp4` / `avoidance_*.mp4`               | Recorded visualization when `--save-video` is set.             |
| `frame_*.jpg`, `depth_*.jpg`, `midas_*.jpg`                    | Screenshots saved via `S` key in testers.                      |
| `camera_calib.npz`, `camera_calib.txt`, `calibration_test.jpg` | Intrinsics, report, undistortion check.                        |
| `checkpoints/*.pt`, `~/.cache/depth_anything_v2/*.pth`         | Model weights downloaded by `init.sh` or `setup_depth_pro.sh`. |

Keep `checkpoints/` under version control (gitignored) but ensure the folder exists before running pipelines.

---

## Troubleshooting

| Issue                            | Resolution                                                                                                                    |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Depth Pro oom / load failure     | `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`, close GUI apps, rerun with `USE_CUDA=False`.                                |
| Chessboard not detected          | Improve lighting, ensure board fully visible, use `utils/test_chessboard_detection.py`, confirm 6×5 inner corners.            |
| Mission Planner shows no MAVLink | Ensure Pixhawk forwards Jetson serial port, or stream via UDP as validation.                                                  |
| Low FPS / thermal throttling     | Switch to `avoidance_midas.py`, disable `--save-video`, set Jetson to 30 W mode (`sudo nvpmodel -m 0 && sudo jetson_clocks`). |
| Depth scale drift                | Re-run calibration (`C` key in MiDaS Pro) or use `size_priors`/`affine_invariant` methods.                                    |
| Disk full                        | Run `bash clean.sh`, prune `/mnt/hdd/docker`, verify `docker system df`.                                                      |
| `/dev/ttyACM0` busy              | Stop MAVROS before launching avoidance script, or change MAVLink endpoint (`--mavlink udp:...`).                              |

---

## Reference Documents

- `docs/camera_calibration_guide.md` — Comprehensive photo-based calibration playbook (capture table, lighting, naming scheme, metric scaling).
- `docs/pixhawk_jetson_connect_instruction.md` — Detailed explanation of serial links, forwarding, Mission Planner inspection, UDP validation.
- `jetson-pixhawk-connect/1.txt` — MAVROS command cheat sheet, telemetry adjustments, ROS 2 topic monitoring.

---

## Authors

**Zhang Jiahe & Yu Yixiang** — 2025–2026 Autonomous Drone Project Lead Developers.

For questions or integration support, reach out to the authors or open an issue in the repository.

---
