"""
Drone obstacle avoidance system - MiDaS Pro version
Based on YOLO11s + MiDaS Small + Metric Depth conversion
Supports three Metric Depth paths:
1. Scene Calibration
2. Object-Size Priors
3. Multi-Frame Geometry
"""

import cv2
import numpy as np
import torch
import time
from pathlib import Path
from collections import deque

from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

try:
    from pymavlink import mavutil
    HAS_MAVLINK = True
except ImportError:
    HAS_MAVLINK = False


# ============ Object-size prior database ============
OBJECT_SIZE_PRIORS = {
    'person': 1.7,      # Average height (meters)
    'bicycle': 1.8,     # Length
    'car': 4.5,         # Length
    'motorcycle': 2.0,
    'bus': 12.0,
    'truck': 8.0,
    'chair': 0.8,       # Height
    'couch': 2.0,
    'dining table': 1.5,
    'bottle': 0.25,
    'cup': 0.12,
    'laptop': 0.35,
    'mouse': 0.08,
    'keyboard': 0.4,
    'cell phone': 0.15,
    'book': 0.25,
    'clock': 0.3,
    'potted plant': 0.5,
    'dog': 0.6,
    'cat': 0.4,
    'bird': 0.15,
}


class MetricDepthConverter:
    """Relative depth → Metric depth converter"""
    
    def __init__(self, method='scene_calib', fx=600, fy=600):
        """
        Args:
            method: 'scene_calib', 'size_priors', 'geometry'
            fx, fy: Camera intrinsics (focal length, pixels)
        """
        self.method = method
        self.fx = fx
        self.fy = fy
        
        # Scene Calibration parameters (need initialization)
        self.calib_a = 10.0  # Default value
        self.calib_b = 0.0
        self.is_calibrated = False
        
        # Multi-Frame Geometry state
        self.prev_frame = None
        self.prev_depth = None
        self.feature_tracker = cv2.SparsePyrLKOpticalFlow_create()
    
    def calibrate_scene(self, depth_rel_roi, known_distance_meters):
        """
        Scene calibration: use object at known distance to fit metric = a/(d_rel) + b
        
        Args:
            depth_rel_roi: MiDaS relative depth ROI (single object region)
            known_distance_meters: Real distance (meters)
        """
        d_rel_mean = np.mean(depth_rel_roi)
        
        # Simplified model: metric ≈ a / d_rel
        # Solution: a = metric * d_rel
        self.calib_a = known_distance_meters * d_rel_mean
        self.calib_b = 0.0
        self.is_calibrated = True
        
        print(f"✅ Scene Calibrated: a={self.calib_a:.2f}, b={self.calib_b:.2f}")
    
    def rel_to_metric_scene(self, depth_rel_map):
        """Scene calibration method: apply metric = a/(d_rel) + b"""
        if not self.is_calibrated:
            print("⚠️  Not calibrated, using default parameters")
        
        # Prevent division by zero
        depth_rel_safe = np.clip(depth_rel_map, 1e-3, None)
        metric_depth = self.calib_a / depth_rel_safe + self.calib_b
        
        return metric_depth
    
    def rel_to_metric_priors(self, depth_rel_map, detections, frame_shape):
        """
        Object-size prior method: distance = (real_size * fx) / pixel_size
        
        Returns:
            dict: {detection_id: metric_distance}
        """
        h, w = frame_shape[:2]
        distances = {}
        
        for i, det in enumerate(detections):
            class_name = det['class_name']
            if class_name not in OBJECT_SIZE_PRIORS:
                continue
            
            real_size = OBJECT_SIZE_PRIORS[class_name]
            x1, y1, x2, y2 = det['bbox']
            
            # Pixel size (take the larger dimension)
            pixel_width = x2 - x1
            pixel_height = y2 - y1
            pixel_size = max(pixel_width, pixel_height)
            
            if pixel_size < 10:  # Too small, skip
                continue
            
            # Pinhole camera model: distance = (real_size * focal_length) / pixel_size
            distance = (real_size * self.fx) / pixel_size
            
            distances[i] = distance
        
        return distances
    
    def rel_to_metric_geometry(self, frame, depth_rel_map):
        """
        Multi-frame geometry method: optical flow + disparity → triangulation (simplified)
        Requires camera motion (translation)
        """
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            self.prev_depth = depth_rel_map.copy()
            return depth_rel_map  # First frame returns relative depth
        
        # Detect feature points (simplified: use goodFeaturesToTrack)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Sparse optical flow
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
        if p0 is None:
            self.prev_frame = frame.copy()
            return depth_rel_map
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
        
        # Filter valid points
        good_old = p0[st == 1]
        good_new = p1[st == 1]
        
        # Compute average displacement (coarse disparity)
        if len(good_old) < 10:
            self.prev_frame = frame.copy()
            return depth_rel_map
        
        disparity = np.linalg.norm(good_new - good_old, axis=1).mean()
        
        # Simplified triangulation: assume camera translation baseline=0.1m
        baseline = 0.1  # meters (needs IMU to provide)
        
        # metric_depth ≈ (baseline * fx) / disparity
        # But this is only a demo, real use needs more complex matching
        
        # Update state
        self.prev_frame = frame.copy()
        self.prev_depth = depth_rel_map.copy()
        
        # Return relative depth (geometry method needs more frames and motion information)
        return depth_rel_map
    
    def convert(self, depth_rel_map, frame=None, detections=None, frame_shape=None):
        """
        Unified interface: relative depth → Metric depth
        
        Returns:
            metric_depth_map (ndarray) or distances_dict (dict)
        """
        if self.method == 'scene_calib':
            return self.rel_to_metric_scene(depth_rel_map)
        
        elif self.method == 'size_priors':
            if detections is None:
                raise ValueError("size_priors requires detections")
            return self.rel_to_metric_priors(depth_rel_map, detections, frame_shape)
        
        elif self.method == 'geometry':
            if frame is None:
                raise ValueError("geometry requires frame")
            return self.rel_to_metric_geometry(frame, depth_rel_map)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")


class DroneAvoidanceSystemPro:
    """Drone obstacle avoidance system - MiDaS Pro version"""
    
    def __init__(self, 
                 metric_method='scene_calib',
                 use_realsense=True,
                 mavlink_port='/dev/ttyACM0',
                 camera_id=0,
                 verbose=True,
                 use_calibration=True):
        """
        Args:
            metric_method: 'scene_calib', 'size_priors', 'geometry'
            use_calibration: Whether to load camera calibration data
        """
        self.verbose = verbose
        self.metric_method = metric_method
        
        print("=" * 60)
        print(f"🚁 Drone Avoidance System Pro (MiDaS + {metric_method})")
        print("=" * 60)
        
        # Load YOLO11s
        print("\n📦 Loading YOLO11s...")
        yolo_path = Path(__file__).parent / 'checkpoints' / 'yolo11s.pt'
        self.yolo = YOLO(str(yolo_path))
        print("✅ YOLO11s loaded")
        
        # Load MiDaS Small
        print("\n📦 Loading MiDaS Small...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas, self.midas_transform = self._load_midas()
        print("✅ MiDaS loaded")
        
        # ============ New: load camera calibration ============
        self.camera_matrix = None
        self.dist_coeffs = None
        self.is_calibrated = False
        
        fx_calib = None
        fy_calib = None
        cx_calib = None
        cy_calib = None
        calib_width = None
        calib_height = None
        
        if use_calibration:
            calib_path = Path(__file__).parent / 'camera_calib.npz'
            if calib_path.exists():
                print(f"\n📐 Loading camera calibration: {calib_path}")
                calib_data = np.load(str(calib_path))
                
                self.camera_matrix = calib_data['camera_matrix']
                self.dist_coeffs = calib_data['dist_coeffs']
                self.is_calibrated = True
                
                calib_width = int(calib_data['img_width'])
                calib_height = int(calib_data['img_height'])
                
                fx_calib = float(self.camera_matrix[0, 0])
                fy_calib = float(self.camera_matrix[1, 1])
                cx_calib = float(self.camera_matrix[0, 2])
                cy_calib = float(self.camera_matrix[1, 2])
                
                print(f"   ✅ Calibration loaded (from {calib_width}x{calib_height}):")
                print(f"      fx={fx_calib:.1f}, fy={fy_calib:.1f}, cx={cx_calib:.1f}, cy={cy_calib:.1f}")
                print(f"      Reprojection error: {calib_data['reprojection_error']:.4f} px")
            else:
                print(f"\n⚠️  Calibration file not found: {calib_path}")
                print("   Using default fx/fy=600")
        
        # ============ Initialize camera ============
        self.use_realsense = False
        self.pipeline = None
        self.cap = None
        
        if use_realsense and HAS_REALSENSE:
            print("\n📷 Trying RealSense...")
            try:
                self.pipeline = self._init_realsense()
                self.use_realsense = True
                print("✅ RealSense initialized")
            except Exception as e:
                print(f"⚠️  RealSense failed: {e}")
                self.use_realsense = False
        
        if not self.use_realsense:
            print(f"\n📷 Initializing camera {camera_id}...")
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, test = self.cap.read()
            if ret:
                self.actual_height, self.actual_width = test.shape[:2]
                print(f"   📷 Runtime resolution: {self.actual_width}x{self.actual_height}")
            else:
                self.actual_width, self.actual_height = 640, 480
            
            # 🔧 Scale focal length to runtime resolution
            if self.is_calibrated and calib_width and calib_height:
                scale_x = self.actual_width / calib_width
                scale_y = self.actual_height / calib_height
                
                fx_final = fx_calib * scale_x
                fy_final = fy_calib * scale_y
                cx_final = cx_calib * scale_x
                cy_final = cy_calib * scale_y
                
                print(f"   🔧 Scaled calibration to runtime resolution:")
                print(f"      fx: {fx_calib:.1f} → {fx_final:.1f} (scale: ×{scale_x:.4f})")
                print(f"      fy: {fy_calib:.1f} → {fy_final:.1f} (scale: ×{scale_y:.4f})")
                
                # Update camera_matrix
                self.camera_matrix = np.array([
                    [fx_final, 0, cx_final],
                    [0, fy_final, cy_final],
                    [0, 0, 1]
                ], dtype=np.float32)
            else:
                # Use default values
                fx_final = 600
                fy_final = 600
            
            print("✅ Camera ready")

        # ============ Initialize Metric Converter ============
        print(f"\n🔧 Initializing Metric Converter ({metric_method})...")
        self.metric_converter = MetricDepthConverter(
            method=metric_method, 
            fx=fx_final,  # Use scaled focal length
            fy=fy_final
        )
        print("✅ Metric Converter ready")
        
        # MAVLink
        self.mavlink = None
        if HAS_MAVLINK:
            try:
                print(f"\n🔌 Connecting MAVLink: {mavlink_port}...")
                self.mavlink = mavutil.mavlink_connection(mavlink_port, baud=57600)
                self.mavlink.wait_heartbeat(timeout=3)
                print("✅ MAVLink connected")
            except Exception as e:
                print(f"⚠️  MAVLink failed: {e}")
        
        # Avoidance parameters
        self.safe_distance = 2.5
        self.danger_distance = 1.5
        self.critical_distance = 0.8
        
        # Stats
        self.frame_count = 0
        self.fps_history = []
        self.last_command = None
        
        print("\n" + "=" * 60)
        print("✅ System ready!")
        print(f"   Metric Method: {metric_method}")
        print(f"   Camera: {'RealSense' if self.use_realsense else 'RGB'}")
        print(f"   MAVLink: {'Connected' if self.mavlink else 'Disabled'}")
        print("=" * 60)
        print("\nKeys: Q-quit | S-save | R-reset | V-verbose | C-calibrate")
        print()
    
    def _load_midas(self):
        """Load MiDaS Small"""
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
            
            print(f"   GPU RAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        return midas, transform
    
    def _init_realsense(self):
        """Initialize RealSense"""
        ctx = rs.context()
        if len(ctx.query_devices()) == 0:
            raise RuntimeError("No RealSense")
        
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        return pipeline

    def undistort_frame(self, frame):
        """Remove lens distortion"""
        if not self.is_calibrated:
            return frame
        
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        dst = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        
        # Optional: crop ROI
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
        
        return dst
    
    def get_frame(self):
        """Get frame + MiDaS relative depth"""
        if self.use_realsense:
            frames = self.pipeline.wait_for_frames()
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                return None, None
            
            frame = np.asanyarray(color.get_data())
            depth_map = np.asanyarray(depth.get_data()) / 1000.0
            return frame, depth_map
        
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None, None

            # Undistort
            if self.is_calibrated:
                frame = self.undistort_frame(frame)
            
            # MiDaS inference
            with torch.no_grad():
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_batch = self.midas_transform(rgb).to(self.device)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                pred = self.midas(input_batch)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=(frame.shape[0], frame.shape[1]),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
                depth_rel = pred.cpu().numpy()
            
            return frame, depth_rel
    
    def process_frame(self, frame, depth_rel_map):
        """Process frame: YOLO + Depth → Metric → decision"""
        results = {}
        
        # YOLO detection
        yolo_res = self.yolo(frame, verbose=False)[0]
        detections = []
        for box in yolo_res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({
                'class_id': int(box.cls[0]),
                'class_name': yolo_res.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
            })
        
        results['detections'] = detections
        results['depth_rel'] = depth_rel_map
        
        # Metric Depth conversion
        if self.metric_method == 'size_priors':
            # Returns {detection_id: distance}
            distances_dict = self.metric_converter.convert(
                depth_rel_map, 
                detections=detections, 
                frame_shape=frame.shape
            )
            results['metric_distances'] = distances_dict
            
            # Build metric_depth_map (for visualization only)
            metric_depth_map = np.zeros_like(depth_rel_map)
            for i, det in enumerate(detections):
                if i in distances_dict:
                    x1, y1, x2, y2 = det['bbox']
                    metric_depth_map[y1:y2, x1:x2] = distances_dict[i]
            
            results['depth_metric'] = metric_depth_map
        
        elif self.metric_method == 'geometry':
            metric_depth_map = self.metric_converter.convert(
                depth_rel_map, 
                frame=frame
            )
            results['depth_metric'] = metric_depth_map
            results['metric_distances'] = {}
        
        else:  # scene_calib
            metric_depth_map = self.metric_converter.convert(depth_rel_map)
            results['depth_metric'] = metric_depth_map
            results['metric_distances'] = {}
        
        # Fusion: compute metric distance for each object
        obstacles = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Prefer precise distance from size_priors
            if self.metric_method == 'size_priors' and i in results['metric_distances']:
                depth_min = results['metric_distances'][i]
                depth_mean = depth_min
            else:
                # Extract from metric_depth_map
                roi = results['depth_metric'][y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                valid = roi[(roi > 0) & ~np.isnan(roi) & (roi < 100)]
                if valid.size == 0:
                    depth_mean = 10.0
                    depth_min = 10.0
                else:
                    depth_mean = float(np.mean(valid))
                    depth_min = float(np.min(valid))
            
            obstacle = {
                **det,
                'depth_mean': depth_mean,
                'depth_min': depth_min,
                'is_critical': depth_min < self.critical_distance,
                'is_dangerous': depth_min < self.danger_distance,
                'is_warning': depth_min < self.safe_distance,
            }
            obstacles.append(obstacle)
        
        results['obstacles'] = obstacles
        
        # Avoidance decision
        cmd = self._generate_command(obstacles, frame.shape)
        results['command'] = cmd
        
        return results
    
    def _generate_command(self, obstacles, shape):
        """Generate avoidance command"""
        cmd = {'action': 'clear', 'direction': None, 'velocity': [0,0,0], 'priority': 0, 'reason': None}
        
        critical = [o for o in obstacles if o['is_critical']]
        dangerous = [o for o in obstacles if o['is_dangerous']]
        
        if critical:
            closest = min(critical, key=lambda x: x['depth_min'])
            cmd['action'] = 'emergency'
            cmd['priority'] = 10
            cmd['reason'] = f"EMERGENCY! {closest['class_name']} @ {closest['depth_min']:.2f}m"
            cmd['direction'] = 'back'
            cmd['velocity'] = [-0.5, 0, 0]
        elif dangerous:
            closest = min(dangerous, key=lambda x: x['depth_min'])
            cmd['action'] = 'avoid'
            cmd['priority'] = 5
            cmd['reason'] = f"Avoid {closest['class_name']} @ {closest['depth_min']:.2f}m"
            cmd['direction'] = 'right'
            cmd['velocity'] = [0, -0.5, 0]
        
        return cmd
    
    def send_mavlink(self, cmd):
        """Send MAVLink"""
        if not self.mavlink or cmd['action'] == 'clear':
            return
        try:
            self.mavlink.mav.set_position_target_local_ned_send(
                0, self.mavlink.target_system, self.mavlink.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED, 0b0000111111000111,
                0, 0, 0, cmd['velocity'][0], cmd['velocity'][1], cmd['velocity'][2],
                0, 0, 0, 0, 0
            )
        except Exception as e:
            print(f"⚠️  MAVLink send failed: {e}")
    
    def print_info(self, results, fps, inf_time):
        """Print detailed info"""
        if not self.verbose:
            return
        
        print(f"\n{'='*70}")
        print(f"🎯 Frame #{self.frame_count:05d} | FPS: {fps:.1f} | Inference: {inf_time*1000:.1f}ms")
        print(f"{'='*70}")
        
        print(f"\n📦 Detections: {len(results['detections'])}")
        for i, d in enumerate(results['detections'][:5], 1):
            print(f"  {i}. {d['class_name']:15s} | Conf: {d['confidence']:.2f}")
        
        print(f"\n⚠️  Obstacles: {len(results['obstacles'])}")
        for i, o in enumerate(results['obstacles'], 1):
            status = "🔴 CRITICAL" if o['is_critical'] else \
                     "🟠 DANGER" if o['is_dangerous'] else \
                     "🟡 WARNING" if o['is_warning'] else "🟢 SAFE"
            
            # Mark used method
            method_tag = ""
            if self.metric_method == 'size_priors' and i-1 in results.get('metric_distances', {}):
                method_tag = " [Prior]"
            elif self.metric_method == 'scene_calib':
                method_tag = " [Calib]"
            elif self.metric_method == 'geometry':
                method_tag = " [Geo]"
            
            print(f"  {i}. {o['class_name']:15s} | {status} | "
                  f"Depth: {o['depth_min']:.2f}m{method_tag} | Conf: {o['confidence']:.2f}")
        
        cmd = results['command']
        print(f"\n🎮 Command:")
        if cmd['action'] == 'clear':
            print(f"  ✅ Path Clear")
        else:
            icon = "🚨" if cmd['action'] == 'emergency' else "⚠️"
            print(f"  {icon} {cmd['action'].upper()}: {cmd['reason']}")
            print(f"  📍 Direction: {cmd['direction']} | Vel: {cmd['velocity']}")
        
        print(f"{'='*70}\n")
    
    def visualize(self, frame, results):
        """Visualization"""
        vis = frame.copy()
        
        # Draw Metric Depth heatmap
        depth_metric = results.get('depth_metric', results['depth_rel'])
        depth_norm = np.clip(depth_metric, 0, 10) / 10 * 255
        depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
        vis = cv2.addWeighted(vis, 0.7, depth_colored, 0.3, 0)
        
        # Draw detection boxes
        for i, obs in enumerate(results['obstacles']):
            x1, y1, x2, y2 = obs['bbox']
            
            color = (0, 0, 255) if obs['is_critical'] else \
                    (0, 165, 255) if obs['is_dangerous'] else \
                    (0, 255, 255) if obs['is_warning'] else (0, 255, 0)
            thickness = 3 if obs['is_critical'] else 2
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Labels
            label = f"{obs['class_name']}"
            depth_label = f"{obs['depth_min']:.2f}m"
            
            # Method tag
            if self.metric_method == 'size_priors' and i in results.get('metric_distances', {}):
                depth_label += " [P]"
            elif self.metric_method == 'scene_calib':
                depth_label += " [C]"
            elif self.metric_method == 'geometry':
                depth_label += " [G]"
            
            cv2.putText(vis, label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(vis, depth_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show command
        cmd = results['command']
        if cmd['action'] != 'clear':
            color = (0, 0, 255) if cmd['action'] == 'emergency' else (0, 165, 255)
            cv2.putText(vis, cmd['reason'], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # FPS
        if self.fps_history:
            avg_fps = np.mean(self.fps_history[-30:])
            cv2.putText(vis, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Method tag
        cv2.putText(vis, f"Method: {self.metric_method}", (10, vis.shape[0]-60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"Frame: {self.frame_count}", (10, vis.shape[0]-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"Objects: {len(results['detections'])}", (10, vis.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis
    
    def calibrate_interactive(self, frame, depth_rel):
        """Interactive scene calibration"""
        print("\n🎯 Scene Calibration Mode")
        print("   1. Place a known-distance object in view")
        print("   2. Draw a box around it (left-click & drag)")
        print("   3. Enter the real distance in meters")
        
        # Copy frame for drawing
        img = frame.copy()
        roi = []
        drawing = False
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, roi, img
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                roi = [x, y, x, y]
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    img = frame.copy()
                    roi[2], roi[3] = x, y
                    cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
            
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                roi[2], roi[3] = x, y
                cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
        
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', mouse_callback)
        
        while True:
            cv2.imshow('Calibration', img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter
                if len(roi) == 4:
                    break
            elif key == 27:  # ESC
                print("   ❌ Calibration cancelled")
                cv2.destroyWindow('Calibration')
                return
        
        cv2.destroyWindow('Calibration')
        
        # Extract ROI depth
        x1, y1, x2, y2 = sorted([roi[0], roi[2]]), sorted([roi[1], roi[3]])
        x1, y1, x2, y2 = x1[0], y1[0], x1[1], y1[1]
        
        roi_depth = depth_rel[y1:y2, x1:x2]
        
        # Input real distance
        try:
            distance_str = input("   Enter real distance (meters): ")
            distance = float(distance_str)
            
            self.metric_converter.calibrate_scene(roi_depth, distance)
            
        except ValueError:
            print("   ❌ Invalid input")
    
    def run(self, save_video=False, output_path='avoidance_midas_pro.mp4', target_fps=20):
        """Run"""
        frame_interval = 1.0 / target_fps
        
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            w = self.actual_width if hasattr(self, 'actual_width') else 640
            h = self.actual_height if hasattr(self, 'actual_height') else 480
            writer = cv2.VideoWriter(output_path, fourcc, target_fps, (w, h))
            print(f"📹 Saving: {output_path}")
        
        print("\n🚀 System running...\n")
        
        try:
            while True:
                t0 = time.time()
                
                frame, depth_rel = self.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                self.frame_count += 1
                
                results = self.process_frame(frame, depth_rel)
                self.send_mavlink(results['command'])
                vis = self.visualize(frame, results)
                
                inf_time = time.time() - t0
                fps = 1.0 / inf_time if inf_time > 0 else 0
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                
                self.print_info(results, fps, inf_time)
                
                cv2.imshow('Drone Avoidance Pro', vis)
                
                if writer:
                    writer.write(vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Exit")
                    break
                elif key == ord('s'):
                    path = f"frame_{self.frame_count:05d}.jpg"
                    cv2.imwrite(path, vis)
                    print(f"💾 Saved: {path}")
                elif key == ord('r'):
                    self.frame_count = 0
                    self.fps_history = []
                    print("🔄 Reset")
                elif key == ord('v'):
                    self.verbose = not self.verbose
                    print(f"🔊 Verbose: {'ON' if self.verbose else 'OFF'}")
                elif key == ord('c'):
                    # Interactive calibration
                    if self.metric_method == 'scene_calib':
                        self.calibrate_interactive(frame, depth_rel)
                    else:
                        print("⚠️  Calibration only for 'scene_calib' method")
                
                elapsed = time.time() - t0
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted")
        
        finally:
            self.cleanup(writer)
    
    def cleanup(self, writer=None):
        """Cleanup"""
        print("\n🧹 Cleaning up...")
        
        if writer:
            writer.release()
        
        if self.use_realsense and self.pipeline:
            self.pipeline.stop()
        elif self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("✅ Shut down")
        print(f"   Frames: {self.frame_count}")
        if self.fps_history:
            print(f"   Avg FPS: {np.mean(self.fps_history):.1f}")
        print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Drone Avoidance Pro (MiDaS + Metric)')
    parser.add_argument('--method', type=str, default='scene_calib',
                       choices=['scene_calib', 'size_priors', 'geometry'],
                       help='Metric depth conversion method')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--no-realsense', action='store_true', help='Disable RealSense')
    parser.add_argument('--save-video', action='store_true', help='Save video')
    parser.add_argument('--mavlink', type=str, default='/dev/ttyACM0', help='MAVLink port')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    parser.add_argument('--no-calibration', action='store_true',  # New parameter
                       help='Do not use camera calibration')
    
    args = parser.parse_args()
    
    system = DroneAvoidanceSystemPro(
        metric_method=args.method,
        use_realsense=not args.no_realsense,
        mavlink_port=args.mavlink,
        camera_id=args.camera,
        verbose=not args.quiet,
        use_calibration=not args.no_calibration  # Pass parameter
    )
    
    system.run(save_video=args.save_video)