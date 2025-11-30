"""
Final version of drone obstacle avoidance system - MiDaS relative depth + Affine-Invariant calibration + YOLO + MAVLink

Features:
- Only keep: MiDaS Small relative depth + Affine-Invariant calibration (depth_metric = s * depth_rel + b)
- Camera: regular USB camera (default), optional RealSense used only as RGB input
- Depth: MiDaS Small, same style as avoidance_midas_pro_v2.py
- Generate body-frame velocity control (BODY_NED) based on nearest obstacle distance
- Direct MAVLink serial connection to Pixhawk

Warning:
- Before arming/sending velocity commands, ALWAYS confirm a safe environment; preferably remove propellers for ground testing first
"""

# cd /home/penguin/capstone/jetson-examples/reComputer/scripts/drone-obstacle-avoidance

# python3 avoidance_midas_pro_v2.py \
#   --method affine_invariant \
#   --camera 0 \
#   --no-realsense \
#   --mavlink /dev/ttyACM0

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import ardupilotmega as mavlink
    HAS_MAVLINK = True
except ImportError:
    HAS_MAVLINK = False


# =========================================
# 1. Affine-Invariant Metric Depth Converter
# =========================================
class MetricDepthConverterAffine:
    """
    Single scheme: Affine-Invariant + scene calibration

    depth_metric = s * depth_rel + b

    - At startup, use a plane/object at a known distance for calibration to estimate s, b
    - For subsequent frames, directly use s, b to map MiDaS relative depth to meters
    """

    def __init__(self):
        self.affine_scale = 1.0
        self.affine_shift = 0.0
        self.is_calibrated = False

    def calibrate_from_roi(self, depth_rel_map, roi_mask, known_distance_m):
        """
        Use an ROI (e.g., a wall or checkerboard) for calibration:
        - Assume the real depth inside the ROI is approximately constant = known_distance_m
        - Use least squares to fit s, b such that s*d_rel + b ≈ known_distance_m
        """
        d_rel = depth_rel_map[roi_mask].astype(np.float32)
        if d_rel.size < 100:
            raise ValueError("Too few pixels in ROI, cannot calibrate")

        # Construct linear equation: s * d_rel + b = D
        A = np.stack([d_rel, np.ones_like(d_rel)], axis=1)  # [N, 2]
        y = np.full_like(d_rel, known_distance_m, dtype=np.float32)

        # Least squares solution
        x, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        s, b = float(x[0]), float(x[1])

        if s <= 0 or not np.isfinite(s) or not np.isfinite(b):
            raise ValueError(f"Unreasonable scale/shift from calibration: s={s}, b={b}")

        self.affine_scale = s
        self.affine_shift = b
        self.is_calibrated = True
        return s, b

    def convert(self, depth_rel_map):
        """
        Convert relative depth to meters; if not calibrated yet, directly return input (relative depth, for debug only)
        """
        depth_rel = depth_rel_map.astype(np.float32)
        if not self.is_calibrated:
            return depth_rel

        depth_m = self.affine_scale * depth_rel + self.affine_shift
        depth_m = np.clip(depth_m, 0.0, 100.0)  # 0~100 m
        return depth_m


# =========================================
# 2. MiDaS Small Inference Wrapper (simplified from Pro V2)
# =========================================
class MidasDepthRunner:
    """
    Use intel-isl/MiDaS MiDaS_small model, output relative depth (H, W)
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas, self.transform = self._load_midas()

    def _load_midas(self):
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        midas.to(self.device)
        midas.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = transforms.small_transform

        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            dummy = torch.randn(1, 3, 256, 256).to(self.device)
            with torch.no_grad():
                _ = midas(dummy)
            torch.cuda.synchronize()

        return midas, transform

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Input BGR image, output relative depth (H, W), larger values mean further away (MiDaS style)
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)

        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pred = self.midas(input_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(frame_bgr.shape[0], frame_bgr.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_rel = pred.cpu().numpy().astype(np.float32)
        # Normalize to [0,1] to stabilize affine fitting
        depth_rel -= depth_rel.min()
        if depth_rel.max() > 1e-6:
            depth_rel /= depth_rel.max()
        return depth_rel


# =========================================
# 3. Main Obstacle Avoidance System
# =========================================
class DroneAvoidanceFinal:
    """
    Final version: MiDaS relative depth + Affine-Invariant calibration + YOLO + MAVLink
    """

    def __init__(self,
                 use_realsense=True,
                 mavlink_port='/dev/ttyACM0',
                 baudrate=115200,
                 camera_id=0,
                 known_distance=1.0,
                 verbose=True):
        self.use_realsense = use_realsense and HAS_REALSENSE
        self.mavlink_port = mavlink_port
        self.baudrate = baudrate
        self.camera_id = camera_id
        self.known_distance = known_distance
        self.verbose = verbose

        # Depth converter
        self.depth_converter = MetricDepthConverterAffine()

        # Camera & depth model
        self.cap = None
        self.pipeline = None
        self.align = None
        self.depth_model = None   # MidasDepthRunner

        # YOLO
        self.det_model = None
        self.class_names = None

        # MAVLink
        self.mav = None

        # Others: monocular "target resolution"
        self.frame_w = 320   # monocular width (left eye)
        self.frame_h = 240   # monocular height

    # ---------- Utils ----------
    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ---------- Initialization ----------
    def _init_camera(self):
        if self.use_realsense:
            if not HAS_REALSENSE:
                raise RuntimeError("pyrealsense2 not installed, cannot use RealSense")
            self._log("[cam] Initializing RealSense...")
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.frame_w, self.frame_h, rs.format.bgr8, 30)
            self.pipeline.start(config)
            align_to = rs.stream.color
            self.align = rs.align(align_to)
        else:
            self._log("[cam] Using regular USB / stereo camera...")
            self.cap = cv2.VideoCapture(self.camera_id)
            # Request 640x240 (left-right concatenated), then crop left 320x240
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            ret, test_frame = self.cap.read()
            if ret:
                th, tw, _ = test_frame.shape
                self._log(f"[cam] Actual frame size: (H={th}, W={tw})")
            else:
                self._log("[cam] Failed to read test frame")

    def _init_models(self):
        self._log("[model] Loading YOLO11s...")
        ckpt_yolo = Path(__file__).parent / "checkpoints" / "yolo11s.pt"
        self.det_model = YOLO(str(ckpt_yolo))
        self.class_names = self.det_model.names

        self._log("[model] Loading MiDaS Small...")
        self.depth_model = MidasDepthRunner()

    def _init_mavlink(self):
        if not HAS_MAVLINK or self.mavlink_port == "dummy":
            self._log("[mavlink] Skip connection (pymavlink not installed or using dummy)")
            return

        self._log(f"[mavlink] Connecting to {self.mavlink_port} @ {self.baudrate} ...")
        self.mav = mavutil.mavlink_connection(
            self.mavlink_port,
            baud=self.baudrate,
            source_system=250,
            source_component=190,
        )

        self._log("[mavlink] Waiting for HEARTBEAT ...")
        hb = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=10)
        if not hb:
            self._log("[mavlink] No HEARTBEAT within 10s, running vision/depth debug only")
            self.mav = None
            return

        self._log(f"[mavlink] HEARTBEAT from sys={hb.get_srcSystem()} comp={hb.get_srcComponent()}")

    # ---------- Frame Acquisition ----------
    def get_frame(self):
        if self.use_realsense:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                return None
            frame = np.asanyarray(color_frame.get_data())
            return frame
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None

            # For your current 640x240 "left-right stereo":
            # Force using only the left half (320x240) as a monocular image
            h, w, _ = frame.shape
            if w == 640 and h == 240:
                mid = w // 2
                frame = frame[:, :mid, :]    # left eye 320x240
                self.frame_w = mid           # update monocular width
                self.frame_h = h             # monocular height 240

            return frame

    def get_frame_and_depth_rel(self):
        frame = self.get_frame()
        if frame is None:
            return None, None
        depth_rel = self.depth_model.infer(frame)
        return frame, depth_rel

    # ---------- Calibration ----------
    def calibrate_affine(self):
        self._log("[calib] Starting Affine calibration, press 'c' to calibrate, 'q' to skip")

        while True:
            frame, depth_rel = self.get_frame_and_depth_rel()
            if frame is None or depth_rel is None:
                continue

            h, w = depth_rel.shape
            cw, ch = int(w * 0.3), int(h * 0.3)
            x1 = w // 2 - cw // 2
            y1 = h // 2 - ch // 2
            x2 = x1 + cw
            y2 = y1 + ch

            roi_mask = np.zeros_like(depth_rel, dtype=bool)
            roi_mask[y1:y2, x1:x2] = True

            vis = frame.copy()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            txt = f"Calib ROI at ~{self.known_distance:.1f}m, press 'c' to calibrate, 'q' to skip"
            cv2.putText(vis, txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("calibration", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                try:
                    s, b = self.depth_converter.calibrate_from_roi(
                        depth_rel, roi_mask, self.known_distance
                    )
                    self._log(f"[calib] Success: scale={s:.4f}, shift={b:.4f}")
                    cv2.destroyWindow("calibration")
                    return
                except Exception as e:
                    self._log(f"[calib] Failed: {e}")

            elif key == ord('q'):
                self._log("[calib] User skipped calibration, using relative depth values")
                cv2.destroyWindow("calibration")
                return

    # ---------- Obstacle Avoidance Logic ----------
    def process_frame(self, frame, depth_metric):
        h, w = depth_metric.shape

        # Front ROI (middle 40% width, upper 50% height)
        fw = int(w * 0.4)
        fh = int(h * 0.5)
        x1 = w // 2 - fw // 2
        x2 = x1 + fw
        y1 = int(h * 0.1)
        y2 = y1 + fh

        roi = depth_metric[y1:y2, x1:x2]
        roi_valid = roi[(roi > 0.01) & np.isfinite(roi)]
        if roi_valid.size == 0:
            min_dist_front = float("inf")
        else:
            min_dist_front = float(np.percentile(roi_valid, 10))

        # Simple BODY_NED velocity strategy
        vx, vy, vz = 0.5, 0.0, 0.0  # default forward 0.5 m/s
        if min_dist_front < 1.0:
            vx = 0.0
            vy = -0.3   # move left
        elif min_dist_front < 2.0:
            vx = 0.2   # slow down

        # YOLO detection (for visualization)
        obstacles = []
        dets = self.det_model(frame, imgsz=640, conf=0.3, verbose=False)[0]
        for box in dets.boxes:
            x1b, y1b, x2b, y2b = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.class_names.get(cls_id, str(cls_id))
            cx = max(0, min(w - 1, (x1b + x2b) // 2))
            cy = max(0, min(h - 1, (y1b + y2b) // 2))
            d = float(depth_metric[cy, cx])
            obstacles.append({
                "bbox": (x1b, y1b, x2b, y2b),
                "cls": cls_name,
                "conf": conf,
                "dist": d,
            })

        cmd = {"vx": vx, "vy": vy, "vz": vz}
        return {
            "obstacles": obstacles,
            "min_dist_front": min_dist_front,
            "cmd": cmd,
            "front_roi": (x1, y1, x2, y2),
        }

    # ---------- MAVLink ----------
    def send_velocity_cmd(self, cmd):
        if self.mav is None:
            return

        vx, vy, vz = cmd["vx"], cmd["vy"], cmd["vz"]

        type_mask = (
            (1 << 0) | (1 << 1) | (1 << 2) |  # ignore position
            (1 << 6) | (1 << 7) | (1 << 8) |  # ignore acceleration
            (1 << 9) | (1 << 10)              # ignore yaw
        )

        self.mav.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            1, 1,
            mavlink.MAV_FRAME_BODY_NED,
            type_mask,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
        )

    # ---------- Visualization ----------
    def visualize(self, frame, depth_metric, results, fps):
        vis = frame.copy()
        h, w = depth_metric.shape

        # Front ROI
        x1, y1, x2, y2 = results["front_roi"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Obstacle boxes
        for obj in results["obstacles"]:
            x1b, y1b, x2b, y2b = obj["bbox"]
            cls_name = obj["cls"]
            conf = obj["conf"]
            dist = obj["dist"]
            cv2.rectangle(vis, (x1b, y1b), (x2b, y2b), (0, 255, 0), 2)
            cv2.putText(vis,
                        f"{cls_name} {conf:.2f} {dist:.1f}m",
                        (x1b, max(0, y1b - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        min_d = results["min_dist_front"]
        cmd = results["cmd"]
        cv2.putText(vis,
                    f"front_min = {min_d:.2f} m, vx={cmd['vx']:.2f}, vy={cmd['vy']:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.putText(vis,
                    f"FPS: {fps:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        # Depth pseudo-color
        depth_show = depth_metric.copy()
        depth_show[~np.isfinite(depth_show)] = 0.0
        depth_show = np.clip(depth_show, 0, 10.0)
        depth_show = (depth_show / 10.0 * 255).astype(np.uint8)
        depth_show = cv2.applyColorMap(depth_show, cv2.COLORMAP_PLASMA)
        depth_show = cv2.resize(depth_show, (w // 3, h // 3))
        vis[0:depth_show.shape[0], w - depth_show.shape[1]:w] = depth_show

        cv2.imshow("avoidance_final", vis)

    # ---------- Main Loop ----------
    def run(self, save_video=False, output_path='avoidance_final.mp4', target_fps=20):
        self._init_camera()
        self._init_models()
        self._init_mavlink()

        self.calibrate_affine()

        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, target_fps,
                                     (self.frame_w, self.frame_h))

        self._log("[run] Starting main loop, press 'q' to exit")
        prev_t = time.time()

        try:
            while True:
                now = time.time()
                dt = now - prev_t
                prev_t = now
                fps = 1.0 / dt if dt > 1e-3 else 0.0

                frame, depth_rel = self.get_frame_and_depth_rel()
                if frame is None or depth_rel is None:
                    continue

                depth_metric = self.depth_converter.convert(depth_rel)
                results = self.process_frame(frame, depth_metric)

                self.send_velocity_cmd(results["cmd"])
                self.visualize(frame, depth_metric, results, fps)
                if writer is not None:
                    writer.write(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            if writer is not None:
                writer.release()
            if self.cap is not None:
                self.cap.release()
            if self.pipeline is not None:
                self.pipeline.stop()
            cv2.destroyAllWindows()
            self._log("[run] Exited")


# =========================================
# 4. CLI
# =========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final Drone Avoidance (MiDaS + Affine-Invariant Only)")
    parser.add_argument('--no-realsense', action='store_true',
                        help='Do not use RealSense, use regular camera only')
    parser.add_argument('--camera', type=int, default=0,
                        help='Regular camera ID')
    parser.add_argument('--mavlink', type=str, default='/dev/ttyACM0',
                        help='MAVLink endpoint, e.g. /dev/ttyACM0, /dev/ttyUSB0, udp:127.0.0.1:14550, or dummy')
    parser.add_argument('--baud', type=int, default=115200,
                        help='Serial baud rate (for serial:/dev/...)')
    parser.add_argument('--known-distance', type=float, default=1.0,
                        help='Real distance (m) of calibration plane/object')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--output', type=str, default='avoidance_final.mp4')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    system = DroneAvoidanceFinal(
        use_realsense=not args.no_realsense,
        mavlink_port=args.mavlink,
        baudrate=args.baud,
        camera_id=args.camera,
        known_distance=args.known_distance,
        verbose=not args.quiet,
    )
    system.run(save_video=args.save_video, output_path=args.output)