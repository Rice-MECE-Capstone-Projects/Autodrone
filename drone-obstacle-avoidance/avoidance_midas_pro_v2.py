"""
Drone obstacle avoidance system - MiDaS Pro V2
Integrates 5 Metric Depth conversion schemes
"""

import cv2
import numpy as np
import torch
import time
from pathlib import Path
from collections import deque
from scipy.optimize import least_squares

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


# ============ Object size prior database ============
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


class MetricDepthConverterV2:
    """Relative depth → Metric depth converter V2 (5 schemes)"""
    
    def __init__(self, method='affine_invariant', fx=600, fy=600, cx=None, cy=None):
        """
        Args:
            method: 'scene_calib', 'size_priors', 'geometry', 
                    'affine_invariant', 'least_squares', 'monocular_sfm', 'neural_refine'
            fx, fy: Camera focal length (pixels)
            cx, cy: Camera principal point (pixels, optional)
        """
        self.method = method
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        # Scene Calibration
        self.calib_a = 10.0
        self.calib_b = 0.0
        self.is_calibrated = False
        
        # Multi-Frame Geometry
        self.prev_frame = None
        self.prev_depth = None
        
        # Affine-Invariant parameters
        self.affine_scale = 1.0
        self.affine_shift = 0.0
        
        # Least-Squares data collection
        self.ls_samples = []  # [(pixel_size, depth_rel, real_distance), ...]
        
        # Monocular SfM
        self.keyframes = deque(maxlen=10)
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Neural Refinement (simplified: linear layer)
        self.neural_weights = None
        self.neural_bias = None
    
    # ============ Original methods (kept unchanged) ============
    def calibrate_scene(self, depth_rel_roi, known_distance_meters):
        """Scene calibration"""
        d_rel_mean = np.mean(depth_rel_roi)
        self.calib_a = known_distance_meters * d_rel_mean
        self.calib_b = 0.0
        self.is_calibrated = True
        print(f"✅ Scene Calibrated: a={self.calib_a:.2f}, b={self.calib_b:.2f}")
    
    def rel_to_metric_scene(self, depth_rel_map):
        """Scene calibration method"""
        depth_rel_safe = np.clip(depth_rel_map, 1e-3, None)
        return self.calib_a / depth_rel_safe + self.calib_b
    
    def rel_to_metric_priors(self, depth_rel_map, detections, frame_shape):
        """Object size prior method"""
        distances = {}
        for i, det in enumerate(detections):
            class_name = det['class_name']
            if class_name not in OBJECT_SIZE_PRIORS:
                continue
            
            real_size = OBJECT_SIZE_PRIORS[class_name]
            x1, y1, x2, y2 = det['bbox']
            pixel_size = max(x2 - x1, y2 - y1)
            
            if pixel_size < 10:
                continue
            
            distance = (real_size * self.fx) / pixel_size
            distances[i] = distance
        
        return distances
    
    # ============ New method 1: Affine-Invariant ============
    def rel_to_metric_affine(self, depth_rel_map, detections=None):
        """
        Affine-invariant depth conversion
        Principle: MiDaS output is affine-invariant, i.e.:
            d_metric = scale * d_rel + shift
        Use objects with known size to fit scale and shift
        """
        if detections is None or len(detections) == 0:
            # Use default parameters
            return self.affine_scale * depth_rel_map + self.affine_shift
        
        # Collect objects with size priors
        samples = []
        for det in detections:
            if det['class_name'] not in OBJECT_SIZE_PRIORS:
                continue
            
            real_size = OBJECT_SIZE_PRIORS[det['class_name']]
            x1, y1, x2, y2 = det['bbox']
            pixel_size = max(x2 - x1, y2 - y1)
            
            if pixel_size < 10:
                continue
            
            # True distance (estimated from camera intrinsics)
            d_metric_true = (real_size * self.fx) / pixel_size
            
            # Corresponding relative depth
            roi_depth_rel = depth_rel_map[y1:y2, x1:x2]
            d_rel_mean = np.mean(roi_depth_rel)
            
            samples.append((d_rel_mean, d_metric_true))
        
        if len(samples) >= 2:
            # Least-squares fit scale and shift
            # d_metric = scale * d_rel + shift
            A = np.array([[d_rel, 1] for d_rel, _ in samples])
            b = np.array([d_metric for _, d_metric in samples])
            
            try:
                params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                self.affine_scale, self.affine_shift = params
                print(f"✅ Affine fitted: scale={self.affine_scale:.2f}, shift={self.affine_shift:.2f}")
            except:
                pass
        
        # Apply transform
        return self.affine_scale * depth_rel_map + self.affine_shift
    
    # ============ New method 2: Least-Squares Regression ============
    def rel_to_metric_least_squares(self, depth_rel_map, detections, frame_shape):
        """
        Least-squares regression depth estimation
        Principle: collect multiple (pixel_size, depth_rel) → real_distance samples
        Fit a nonlinear model: distance = (real_size * fx) / pixel_size
        """
        # Collect new samples
        for det in detections:
            if det['class_name'] not in OBJECT_SIZE_PRIORS:
                continue
            
            real_size = OBJECT_SIZE_PRIORS[det['class_name']]
            x1, y1, x2, y2 = det['bbox']
            pixel_size = max(x2 - x1, y2 - y1)
            
            if pixel_size < 10:
                continue
            
            roi_depth_rel = depth_rel_map[y1:y2, x1:x2]
            d_rel_mean = np.mean(roi_depth_rel)
            
            # True distance
            d_metric = (real_size * self.fx) / pixel_size
            
            self.ls_samples.append((pixel_size, d_rel_mean, d_metric))
        
        # Keep sample size reasonable
        if len(self.ls_samples) > 100:
            self.ls_samples = self.ls_samples[-100:]
        
        # If samples are insufficient, fall back to simple conversion
        if len(self.ls_samples) < 5:
            return self.rel_to_metric_affine(depth_rel_map, detections)
        
        # Fit nonlinear model: d_metric = a / d_rel + b
        def residual(params):
            a, b = params
            errors = []
            for pixel_size, d_rel, d_metric_true in self.ls_samples:
                d_metric_pred = a / d_rel + b
                errors.append(d_metric_pred - d_metric_true)
            return errors
        
        try:
            result = least_squares(residual, [10.0, 0.0], method='lm')
            a, b = result.x
            
            depth_rel_safe = np.clip(depth_rel_map, 1e-3, None)
            metric_depth = a / depth_rel_safe + b
            
            print(f"✅ Least-Squares: a={a:.2f}, b={b:.2f}, samples={len(self.ls_samples)}")
            
            return metric_depth
        
        except:
            # Fall back to affine transform
            return self.rel_to_metric_affine(depth_rel_map, detections)
    
    # ============ New method 3: Monocular SfM ============
    def rel_to_metric_sfm(self, frame, depth_rel_map):
        """
        Monocular Structure from Motion
        Principle: use camera motion, feature matching and triangulation to estimate true scale
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if des is None or len(kp) < 20:
            return depth_rel_map  # Not enough features, return original
        
        # Store keyframe
        self.keyframes.append({
            'frame': gray.copy(),
            'depth': depth_rel_map.copy(),
            'kp': kp,
            'des': des
        })
        
        if len(self.keyframes) < 2:
            return depth_rel_map
        
        # Match the two most recent frames
        prev_kf = self.keyframes[-2]
        curr_kf = self.keyframes[-1]
        
        matches = self.bf_matcher.match(prev_kf['des'], curr_kf['des'])
        matches = sorted(matches, key=lambda x: x.distance)[:50]
        
        if len(matches) < 10:
            return depth_rel_map
        
        # Extract matched points
        pts1 = np.float32([prev_kf['kp'][m.queryIdx].pt for m in matches])
        pts2 = np.float32([curr_kf['kp'][m.trainIdx].pt for m in matches])
        
        # Estimate essential matrix (requires intrinsics)
        if self.cx is None or self.cy is None:
            # Use image center
            h, w = frame.shape[:2]
            cx, cy = w / 2, h / 2
        else:
            cx, cy = self.cx, self.cy
        
        K = np.array([
            [self.fx, 0, cx],
            [0, self.fy, cy],
            [0, 0, 1]
        ])
        
        try:
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
            
            # Triangulation (simplified)
            # In theory can obtain true scale, but needs IMU to provide absolute motion
            # Here is only a demonstration framework
            
            # In practical applications we need:
            # 1. Multi-view geometry
            # 2. Bundle Adjustment
            # 3. Loop Closure
            
            print(f"✅ SfM: Matched {len(matches)} features, R={R.shape}, t={t.shape}")
            
        except:
            pass
        
        # Return relative depth (full implementation needs more work)
        return depth_rel_map
    
    # ============ New method 4: Neural Refinement ============
    def rel_to_metric_neural(self, depth_rel_map, detections=None):
        """
        Neural-network-based depth refinement
        Principle: use a simple linear layer to learn mapping d_rel → d_metric
        Need to collect ground truth labels (can use outputs of other methods as pseudo labels)
        """
        # Initialize weights (if not initialized)
        if self.neural_weights is None:
            # Use affine transform initial values
            self.neural_weights = self.affine_scale
            self.neural_bias = self.affine_shift
        
        # Simplified: directly apply linear transform (can be trained in practice)
        metric_depth = self.neural_weights * depth_rel_map + self.neural_bias
        
        # TODO: Online learning (if real labels exist)
        # Can use outputs of other methods as supervision
        
        return metric_depth
    
    # ============ Unified interface ============
    def convert(self, depth_rel_map, frame=None, detections=None, frame_shape=None):
        """Unified conversion interface"""
        
        if self.method == 'scene_calib':
            return self.rel_to_metric_scene(depth_rel_map)
        
        elif self.method == 'size_priors':
            if detections is None:
                raise ValueError("size_priors requires detections")
            return self.rel_to_metric_priors(depth_rel_map, detections, frame_shape)
        
        elif self.method == 'affine_invariant':
            return self.rel_to_metric_affine(depth_rel_map, detections)
        
        elif self.method == 'least_squares':
            if detections is None:
                detections = []
            return self.rel_to_metric_least_squares(depth_rel_map, detections, frame_shape)
        
        elif self.method == 'monocular_sfm':
            if frame is None:
                raise ValueError("monocular_sfm requires frame")
            return self.rel_to_metric_sfm(frame, depth_rel_map)
        
        elif self.method == 'neural_refine':
            return self.rel_to_metric_neural(depth_rel_map, detections)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")


class DroneAvoidanceSystemProV2:
    """Drone obstacle avoidance system - MiDaS Pro V2 (supports 5 schemes)"""
    
    def __init__(self, 
                 metric_method='affine_invariant',
                 use_realsense=True,
                 mavlink_port='/dev/ttyACM0',
                 camera_id=0,
                 verbose=True,
                 use_calibration=True):
        
        self.verbose = verbose
        self.metric_method = metric_method
        
        print("=" * 70)
        print(f"🚁 Drone Avoidance System Pro V2 (MiDaS + {metric_method})")
        print("=" * 70)
        
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
        
        # ============ Load camera calibration ============
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
            
            # Scale focal length
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
                
                self.camera_matrix = np.array([
                    [fx_final, 0, cx_final],
                    [0, fy_final, cy_final],
                    [0, 0, 1]
                ], dtype=np.float32)
            else:
                fx_final = 600
                fy_final = 600
                cx_final = self.actual_width / 2
                cy_final = self.actual_height / 2
            
            print("✅ Camera ready")
        
        # ============ Initialize Metric Converter V2 ============
        print(f"\n🔧 Initializing Metric Converter V2 ({metric_method})...")
        self.metric_converter = MetricDepthConverterV2(
            method=metric_method,
            fx=fx_final,
            fy=fy_final,
            cx=cx_final,
            cy=cy_final
        )
        print("✅ Metric Converter V2 ready")
        
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
        
        # Obstacle avoidance parameters
        self.safe_distance = 2.5
        self.danger_distance = 1.5
        self.critical_distance = 0.8
        
        # Statistics
        self.frame_count = 0
        self.fps_history = []
        self.last_command = None
        
        print("\n" + "=" * 70)
        print("✅ System ready!")
        print(f"   Method: {metric_method}")
        print(f"   Camera: {'RealSense' if self.use_realsense else 'RGB'}")
        print(f"   MAVLink: {'Connected' if self.mavlink else 'Disabled'}")
        print("=" * 70)
        print("\nKeys: Q-quit | S-save | R-reset | V-verbose")
        print()
    
    def _load_midas(self):
        """Load MiDaS"""
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
        """Undistort frame"""
        if not self.is_calibrated:
            return frame
        
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        dst = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        return dst
    
    def get_frame(self):
        """Get frame"""
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
        """Process frame"""
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
        if self.metric_method in ['size_priors']:
            distances_dict = self.metric_converter.convert(
                depth_rel_map,
                detections=detections,
                frame_shape=frame.shape
            )
            results['metric_distances'] = distances_dict
            
            metric_depth_map = np.zeros_like(depth_rel_map)
            for i, det in enumerate(detections):
                if i in distances_dict:
                    x1, y1, x2, y2 = det['bbox']
                    metric_depth_map[y1:y2, x1:x2] = distances_dict[i]
            
            results['depth_metric'] = metric_depth_map
        
        elif self.metric_method in ['affine_invariant', 'least_squares', 'neural_refine']:
            # These methods return a full metric depth map
            metric_depth_map = self.metric_converter.convert(
                depth_rel_map,
                frame=frame,
                detections=detections,
                frame_shape=frame.shape
            )
            results['depth_metric'] = metric_depth_map
            results['metric_distances'] = {}
        
        elif self.metric_method == 'monocular_sfm':
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
        
        # Fusion
        obstacles = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if self.metric_method == 'size_priors' and i in results['metric_distances']:
                depth_min = results['metric_distances'][i]
                depth_mean = depth_min
            else:
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
        
        # Obstacle avoidance decision
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
        """Print info"""
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
            
            method_tag = f" [{self.metric_method[:3].upper()}]"
            
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
        
        depth_metric = results.get('depth_metric', results['depth_rel'])
        depth_norm = np.clip(depth_metric, 0, 10) / 10 * 255
        depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
        vis = cv2.addWeighted(vis, 0.7, depth_colored, 0.3, 0)
        
        for i, obs in enumerate(results['obstacles']):
            x1, y1, x2, y2 = obs['bbox']
            
            color = (0, 0, 255) if obs['is_critical'] else \
                    (0, 165, 255) if obs['is_dangerous'] else \
                    (0, 255, 255) if obs['is_warning'] else (0, 255, 0)
            thickness = 3 if obs['is_critical'] else 2
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{obs['class_name']}"
            depth_label = f"{obs['depth_min']:.2f}m [{self.metric_method[:3]}]"
            
            cv2.putText(vis, label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(vis, depth_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cmd = results['command']
        if cmd['action'] != 'clear':
            color = (0, 0, 255) if cmd['action'] == 'emergency' else (0, 165, 255)
            cv2.putText(vis, cmd['reason'], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if self.fps_history:
            avg_fps = np.mean(self.fps_history[-30:])
            cv2.putText(vis, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(vis, f"Method: {self.metric_method}", (10, vis.shape[0]-60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"Frame: {self.frame_count}", (10, vis.shape[0]-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"Objects: {len(results['detections'])}", (10, vis.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis
    
    def run(self, save_video=False, output_path='avoidance_midas_pro_v2.mp4', target_fps=20):
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
                
                cv2.imshow('Drone Avoidance Pro V2', vis)
                
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
        
        print("\n" + "=" * 70)
        print("✅ Shut down")
        print(f"   Frames: {self.frame_count}")
        if self.fps_history:
            print(f"   Avg FPS: {np.mean(self.fps_history):.1f}")
        print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Drone Avoidance Pro V2')
    parser.add_argument('--method', type=str, default='affine_invariant',
                       choices=['scene_calib', 'size_priors', 
                               'affine_invariant', 'least_squares', 
                               'monocular_sfm', 'neural_refine'],
                       help='Metric depth conversion method')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--no-realsense', action='store_true')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--mavlink', type=str, default='/dev/ttyACM0')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--no-calibration', action='store_true')
    
    args = parser.parse_args()
    
    system = DroneAvoidanceSystemProV2(
        metric_method=args.method,
        use_realsense=not args.no_realsense,
        mavlink_port=args.mavlink,
        camera_id=args.camera,
        verbose=not args.quiet,
        use_calibration=not args.no_calibration
    )
    
    system.run(save_video=args.save_video)