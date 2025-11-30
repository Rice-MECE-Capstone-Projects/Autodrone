"""
Drone obstacle avoidance system - MiDaS version
Based on YOLO11s + MiDaS Small
Supports RealSense depth camera or regular RGB camera
"""

import cv2
import numpy as np
import torch
import sys
import os
import time
from pathlib import Path

from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    print("⚠️ RealSense SDK not installed, will use regular camera")

try:
    from pymavlink import mavutil
    HAS_MAVLINK = True
except ImportError:
    HAS_MAVLINK = False
    print("⚠️ PyMAVLink not installed, avoidance commands will only be displayed")


class DroneAvoidanceSystem:
    """Drone obstacle avoidance system - MiDaS version"""
    
    def __init__(self, 
                 use_realsense=True,
                 use_depth_model=True,
                 mavlink_port='/dev/ttyACM0',
                 camera_id=0,
                 verbose=True):
        """
        Initialize obstacle avoidance system
        
        Args:
            use_realsense: Whether to try using RealSense depth camera
            use_depth_model: Whether to use MiDaS depth model
            mavlink_port: MAVLink serial port
            camera_id: regular camera ID
            verbose: Whether to print detailed info per frame
        """
        self.verbose = verbose
        self.use_depth_model = use_depth_model
        
        print("=" * 60)
        print("🚁 Drone Obstacle Avoidance System v2.0 (MiDaS)")
        print("=" * 60)
        
        # Load YOLO11s
        print("\n📦 Loading YOLO11s model...")
        yolo_model_path = Path(__file__).parent / 'checkpoints' / 'yolo11s.pt'
        self.yolo = YOLO(str(yolo_model_path))
        print("✅ YOLO11s loaded")
        
        # Load MiDaS Small depth model
        self.midas = None
        self.midas_transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if use_depth_model:
            print("\n📦 Loading MiDaS Small model...")
            self.midas, self.midas_transform = self._load_midas_model()
            print("✅ MiDaS Small loaded")
        
        # Initialize camera (with RealSense detection)
        self.use_realsense = False
        self.pipeline = None
        self.cap = None
        
        if use_realsense and HAS_REALSENSE:
            print("\n📷 Attempting to initialize RealSense depth camera...")
            try:
                self.pipeline = self._init_realsense()
                self.use_realsense = True
                print("✅ RealSense initialized")
            except Exception as e:
                print(f"⚠️ RealSense initialization failed: {e}")
                print("   Falling back to regular camera...")
                self.use_realsense = False
        
        # If RealSense fails or not enabled, use regular camera
        if not self.use_realsense:
            print(f"\n📷 Initializing camera (ID: {camera_id})...")
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Get actual resolution
            ret, test_frame = self.cap.read()
            if ret:
                self.actual_height, self.actual_width = test_frame.shape[:2]
                print(f"   📷 Actual resolution: {self.actual_width}x{self.actual_height}")
            else:
                self.actual_width, self.actual_height = 640, 480
                print("   ⚠️ Failed to read test frame, using default resolution")
            
            print("✅ Camera initialized")
        
        # Initialize MAVLink
        self.mavlink = None
        if HAS_MAVLINK:
            try:
                print(f"\n🔌 Connecting to Pixhawk: {mavlink_port}...")
                self.mavlink = mavutil.mavlink_connection(mavlink_port, baud=57600)
                self.mavlink.wait_heartbeat(timeout=3)
                print("✅ Pixhawk connected")
            except Exception as e:
                print(f"⚠️ Pixhawk connection failed: {e}")
                self.mavlink = None
        
        # Avoidance parameters
        self.safe_distance = 2.5   # Safe distance (meters)
        self.danger_distance = 1.5  # Danger distance (meters)
        self.critical_distance = 0.8  # Critical distance (meters)
        
        # Statistics
        self.frame_count = 0
        self.fps_history = []
        self.last_command = None
        
        print("\n" + "=" * 60)
        print("✅ System initialization completed!")
        print(f"   Camera Mode: {'RealSense Depth' if self.use_realsense else 'RGB + MiDaS'}")
        print(f"   Depth Model: {'Enabled' if use_depth_model else 'Disabled'}")
        print(f"   MAVLink: {'Connected' if self.mavlink else 'Not connected'}")
        print("=" * 60)
        print("\nKey bindings:")
        print("  q - Exit system")
        print("  s - Save current frame")
        print("  r - Reset statistics")
        print("  v - Toggle verbose output")
        print("\n")
    
    def _load_midas_model(self):
        """Load MiDaS Small model"""
        # Load MiDaS Small (auto download)
        model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        midas.to(self.device)
        midas.eval()
        
        # Load preprocessing transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = midas_transforms.small_transform
        
        if torch.cuda.is_available():
            # CUDA optimization
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            print("   ✅ cuDNN optimization enabled for MiDaS")
            
            # Warm up model
            print("   🔍 Warming up MiDaS model...")
            dummy = torch.randn(1, 3, 256, 256).to(self.device)
            with torch.no_grad():
                _ = midas(dummy)
            torch.cuda.synchronize()
            
            mem_used = torch.cuda.memory_allocated() / 1e9
            print(f"   ✅ MiDaS ready (GPU RAM: {mem_used:.2f} GB)")
        
        return midas, transform
    
    def _init_realsense(self):
        """Initialize RealSense camera (may raise exception)"""
        # Check RealSense devices
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            raise RuntimeError("No RealSense device connected")
        
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        
        return pipeline
    
    def get_frame(self):
        """Get one frame and depth map"""
        if self.use_realsense:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) / 1000.0  # convert to meters
            
            return color_image, depth_image
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None, None
            
            # Use MiDaS to estimate depth
            if self.midas and self.use_depth_model:
                with torch.no_grad():
                    # Convert to RGB
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Preprocess
                    input_batch = self.midas_transform(rgb).to(self.device)
                    
                    # Inference
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    prediction = self.midas(input_batch)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # Resize to original size
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(frame.shape[0], frame.shape[1]),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                    
                    depth = prediction.cpu().numpy()
                    
                    # Normalize to meters (MiDaS outputs relative depth; scale it)
                    # Assume scene depth range is 0-10 m
                    depth_image = (depth - depth.min()) / (depth.max() - depth.min()) * 10.0
            else:
                # Placeholder depth map
                depth_image = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            
            return frame, depth_image
    
    def process_frame(self, frame, depth_map):
        """
        Process single frame
        
        Returns:
            dict: containing detection results, obstacle info, and avoidance command
        """
        results = {}
        
        # Ensure depth map and frame are same size
        if depth_map.shape[:2] != frame.shape[:2]:
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        
        # 1. YOLO detection
        yolo_results = self.yolo(frame, verbose=False)[0]
        detections = []
        
        for box in yolo_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            det = {
                'class_id': int(box.cls[0]),
                'class_name': yolo_results.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
            }
            detections.append(det)
        
        results['detections'] = detections
        results['depth_map'] = depth_map
        
        # 2. Fusion: compute depth for each object
        obstacles = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Boundary check
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Extract ROI depth
            roi_depth = depth_map[y1:y2, x1:x2]
            if roi_depth.size == 0:
                continue
            
            # Filter invalid depths
            valid_depth = roi_depth[(roi_depth > 0) & ~np.isnan(roi_depth)]
            if valid_depth.size == 0:
                depth_mean = 10.0
                depth_min = 10.0
            else:
                depth_mean = float(np.mean(valid_depth))
                depth_min = float(np.min(valid_depth))
            
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
        
        # 3. Generate avoidance command
        avoidance_cmd = self._generate_avoidance_command(obstacles, frame.shape)
        results['avoidance_command'] = avoidance_cmd
        
        return results
    
    def _generate_avoidance_command(self, obstacles, frame_shape):
        """Generate avoidance command"""
        cmd = {
            'action': 'clear',
            'direction': None,
            'velocity': [0, 0, 0],
            'priority': 0,
            'reason': None
        }
        
        critical = [obs for obs in obstacles if obs['is_critical']]
        dangerous = [obs for obs in obstacles if obs['is_dangerous']]
        
        if critical:
            closest = min(critical, key=lambda x: x['depth_min'])
            cmd.update(self._calculate_avoidance_direction(closest, frame_shape, obstacles))
            cmd['action'] = 'emergency'
            cmd['priority'] = 10
            cmd['reason'] = f"EMERGENCY! {closest['class_name']} @ {closest['depth_min']:.2f}m"
            
        elif dangerous:
            closest = min(dangerous, key=lambda x: x['depth_min'])
            cmd.update(self._calculate_avoidance_direction(closest, frame_shape, obstacles))
            cmd['action'] = 'avoid'
            cmd['priority'] = 5
            cmd['reason'] = f"Avoidance: {closest['class_name']} @ {closest['depth_min']:.2f}m"
        
        return cmd
    
    def _calculate_avoidance_direction(self, obstacle, frame_shape, all_obstacles):
        """Calculate avoidance direction"""
        x1, y1, x2, y2 = obstacle['bbox']
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        
        height, width = frame_shape[:2]
        
        # Check safety of different directions
        safety_zones = {
            'left': self._is_zone_safe(0, width//3, all_obstacles),
            'center': self._is_zone_safe(width//3, 2*width//3, all_obstacles),
            'right': self._is_zone_safe(2*width//3, width, all_obstacles),
        }
        
        # Choose avoidance direction intelligently
        if x_center < width / 2:
            if safety_zones['right']:
                direction = 'right'
                velocity = [0, -0.5, 0]
            elif safety_zones['left']:
                direction = 'left'
                velocity = [0, 0.5, 0]
            else:
                if y_center < height / 2:
                    direction = 'up'
                    velocity = [0, 0, -0.3]
                else:
                    direction = 'back'
                    velocity = [-0.5, 0, 0]
        else:
            if safety_zones['left']:
                direction = 'left'
                velocity = [0, 0.5, 0]
            elif safety_zones['right']:
                direction = 'right'
                velocity = [0, -0.5, 0]
            else:
                if y_center < height / 2:
                    direction = 'up'
                    velocity = [0, 0, -0.3]
                else:
                    direction = 'back'
                    velocity = [-0.5, 0, 0]
        
        return {'direction': direction, 'velocity': velocity}

    def _is_zone_safe(self, x_start, x_end, obstacles):
        """Check if a given horizontal zone is safe"""
        for obs in obstacles:
            x1, _, x2, _ = obs['bbox']
            x_center = (x1 + x2) / 2
            
            if x_start <= x_center <= x_end:
                if obs['is_dangerous'] or obs['is_critical']:
                    return False
        
        return True
    
    def send_mavlink_command(self, cmd):
        """Send MAVLink command"""
        if not self.mavlink or cmd['action'] == 'clear':
            return
        
        try:
            self.mavlink.mav.set_position_target_local_ned_send(
                0,
                self.mavlink.target_system,
                self.mavlink.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111000111,
                0, 0, 0,
                cmd['velocity'][0], cmd['velocity'][1], cmd['velocity'][2],
                0, 0, 0,
                0, 0
            )
        except Exception as e:
            print(f"⚠️ MAVLink send failed: {e}")
    
    def print_frame_info(self, results, fps, inference_time):
        """Print per-frame detailed info to terminal"""
        if not self.verbose:
            return
        
        print(f"\n{'='*70}")
        print(f"🎯 Frame #{self.frame_count:05d} | FPS: {fps:.1f} | Inference: {inference_time*1000:.1f}ms")
        print(f"{'='*70}")
        
        print(f"\n📦 Detections: {len(results['detections'])}")
        if results['detections']:
            for i, det in enumerate(results['detections'][:5], 1):
                print(f"  {i}. {det['class_name']:15s} | Conf: {det['confidence']:.2f}")
        
        print(f"\n⚠️  Obstacles: {len(results['obstacles'])}")
        if results['obstacles']:
            for i, obs in enumerate(results['obstacles'], 1):
                status = "🔴 CRITICAL" if obs['is_critical'] else \
                         "🟠 DANGER" if obs['is_dangerous'] else \
                         "🟡 WARNING" if obs['is_warning'] else "🟢 SAFE"
                print(f"  {i}. {obs['class_name']:15s} | {status} | "
                      f"Depth: {obs['depth_min']:.2f}m | Conf: {obs['confidence']:.2f}")
        
        cmd = results['avoidance_command']
        print(f"\n🎮 Command:")
        if cmd['action'] == 'clear':
            print(f"  ✅ Path Clear - No action needed")
        else:
            action_icon = "🚨" if cmd['action'] == 'emergency' else "⚠️"
            print(f"  {action_icon} Action: {cmd['action'].upper()}")
            print(f"  📍 Direction: {cmd['direction']}")
            print(f"  🎚️  Velocity: [{cmd['velocity'][0]:.2f}, {cmd['velocity'][1]:.2f}, {cmd['velocity'][2]:.2f}]")
            print(f"  📝 Reason: {cmd['reason']}")
            
            if self.last_command != cmd['action']:
                print(f"  ⚡ Command changed from [{self.last_command}] to [{cmd['action']}]")
                self.last_command = cmd['action']
        
        print(f"{'='*70}\n")
    
    def visualize(self, frame, results):
        """Visualize results"""
        vis = frame.copy()
        
        # Draw depth map (heatmap)
        depth_colored = cv2.applyColorMap(
            (results['depth_map'] * 25.5).astype(np.uint8),
            cv2.COLORMAP_MAGMA
        )
        vis = cv2.addWeighted(vis, 0.7, depth_colored, 0.3, 0)
        
        # Draw detection boxes
        for obs in results['obstacles']:
            x1, y1, x2, y2 = obs['bbox']
            
            if obs['is_critical']:
                color = (0, 0, 255)
                thickness = 3
            elif obs['is_dangerous']:
                color = (0, 165, 255)
                thickness = 2
            elif obs['is_warning']:
                color = (0, 255, 255)
                thickness = 2
            else:
                color = (0, 255, 0)
                thickness = 1
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{obs['class_name']}"
            depth_label = f"{obs['depth_min']:.2f}m"
            conf_label = f"{obs['confidence']:.2f}"
            
            cv2.putText(vis, label, (x1, y1 - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(vis, depth_label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(vis, conf_label, (x2 - 50, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Show avoidance command
        cmd = results['avoidance_command']
        if cmd['action'] != 'clear':
            cmd_color = (0, 0, 255) if cmd['action'] == 'emergency' else (0, 165, 255)
            cv2.putText(vis, f"{cmd['reason']}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, cmd_color, 2)
            cv2.putText(vis, f"Direction: {cmd['direction']}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cmd_color, 2)
        
        # Show FPS and stats
        if self.fps_history:
            avg_fps = np.mean(self.fps_history[-30:])
            cv2.putText(vis, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(vis, f"Frame: {self.frame_count}", (10, vis.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"Objects: {len(results['detections'])}", (10, vis.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis
    
    def run(self, save_video=False, output_path='drone_avoidance_midas.mp4', target_fps=20):
        """Run obstacle avoidance system"""
        frame_interval = 1.0 / target_fps
        
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Use actual resolution
            width = self.actual_width if hasattr(self, 'actual_width') else 640
            height = self.actual_height if hasattr(self, 'actual_height') else 480
            video_writer = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
            print(f"📹 Saving video to: {output_path}")
        
        print("\n🚀 System running...\n")
        
        try:
            while True:
                start_time = time.time()
                
                frame, depth_map = self.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                self.frame_count += 1
                
                results = self.process_frame(frame, depth_map)
                self.send_mavlink_command(results['avoidance_command'])
                vis_frame = self.visualize(frame, results)
                
                inference_time = time.time() - start_time
                fps = 1.0 / inference_time if inference_time > 0 else 0
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                
                self.print_frame_info(results, fps, inference_time)
                
                cv2.imshow('Drone Obstacle Avoidance (MiDaS)', vis_frame)
                
                if video_writer:
                    video_writer.write(vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 User exit")
                    break
                elif key == ord('s'):
                    save_path = f"frame_{self.frame_count:05d}.jpg"
                    cv2.imwrite(save_path, vis_frame)
                    print(f"💾 Frame saved: {save_path}")
                elif key == ord('r'):
                    self.frame_count = 0
                    self.fps_history = []
                    self.last_command = None
                    print("🔄 Statistics reset")
                elif key == ord('v'):
                    self.verbose = not self.verbose
                    status = "ON" if self.verbose else "OFF"
                    print(f"🔊 Verbose output: {status}")
                
                elapsed = time.time() - start_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted")
        
        finally:
            self.cleanup(video_writer)
    
    def cleanup(self, video_writer=None):
        """Clean up resources"""
        print("\n🧹 Cleaning up resources...")
        
        if video_writer:
            video_writer.release()
        
        if self.use_realsense and self.pipeline:
            self.pipeline.stop()
        elif self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("✅ System safely shut down")
        print(f"📊 Total frames processed: {self.frame_count}")
        if self.fps_history:
            print(f"📊 Average FPS: {np.mean(self.fps_history):.1f}")
        print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Drone Obstacle Avoidance System (MiDaS)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--no-realsense', action='store_true', help='Do not use RealSense')
    parser.add_argument('--no-depth-model', action='store_true', help='Do not use MiDaS depth model')
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    parser.add_argument('--mavlink', type=str, default='/dev/ttyACM0', help='MAVLink serial port')
    parser.add_argument('--quiet', action='store_true', help='Disable verbose terminal output')
    
    args = parser.parse_args()
    
    # Initialize system
    system = DroneAvoidanceSystem(
        use_realsense=not args.no_realsense,
        use_depth_model=not args.no_depth_model,
        mavlink_port=args.mavlink,
        camera_id=args.camera,
        verbose=not args.quiet
    )
    
    # Run
    system.run(save_video=args.save_video)