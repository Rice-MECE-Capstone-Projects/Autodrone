"""
Depth Anything V2 real-time depth estimation - Jetson Orin optimized
- Force CUDA acceleration
- TensorRT acceleration (optional)
- High-performance inference
"""

import cv2
import torch
import numpy as np
import time
import psutil
from pathlib import Path
import sys
from threading import Thread
from queue import Queue

# Depth Anything V2
depth_v2_path = Path.home() / '.cache/depth_anything_v2/Depth-Anything-V2'
sys.path.insert(0, str(depth_v2_path))
from depth_anything_v2.dpt import DepthAnythingV2

# ============ Configuration ============
CAMERA_ID = 0
WIDTH, HEIGHT = 640, 480
USE_CUDA = torch.cuda.is_available()

print("🚀 Jetson Orin Depth Detection System")
print(f"   CUDA Available: {USE_CUDA}")
if USE_CUDA:
    print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============ Video capture thread ============
class VideoCapture:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.q = Queue(maxsize=2)
        self.stopped = False
        
        # Warm up
        for _ in range(5):
            self.cap.read()
        
        Thread(target=self._reader, daemon=True).start()
    
    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if not self.q.full():
                self.q.put(frame)
    
    def read(self):
        return self.q.get() if not self.q.empty() else None
    
    def stop(self):
        self.stopped = True
        self.cap.release()

# ============ Load model ============
print("📦 Loading model...")
device = 'cuda:0' if USE_CUDA else 'cpu'
print(f"   Target device: {device}")

model = DepthAnythingV2(
    encoder='vits',
    features=64,
    out_channels=[48, 96, 192, 384]
)

checkpoint = Path.home() / '.cache/depth_anything_v2/depth_anything_v2_vits.pth'
model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
model = model.to(device)
model.eval()

# Optimization settings
if USE_CUDA:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    print("   ✅ cuDNN optimization enabled")
    
    # Test CUDA inference
    print("   🔍 Testing CUDA inference...")
    dummy_input = torch.randn(480, 640, 3).numpy()
    with torch.no_grad():
        _ = model.infer_image(dummy_input)
    torch.cuda.synchronize()
    print("   ✅ CUDA inference working")

print("✅ Model ready")

# ============ Initialize camera ============
print("📹 Starting camera...")
cam = VideoCapture(CAMERA_ID)
time.sleep(1)
print("✅ Camera ready\n")

# ============ GPU monitor ============
def get_gpu_stats():
    """Get detailed GPU statistics"""
    stats = {'load': 0, 'mem_used': 0, 'mem_total': 0}
    
    # GPU load
    try:
        with open('/sys/devices/gpu.0/load', 'r') as f:
            stats['load'] = int(f.read().strip()) / 10
    except:
        pass
    
    # GPU memory (PyTorch)
    if USE_CUDA:
        stats['mem_used'] = torch.cuda.memory_allocated() / 1e9
        stats['mem_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return stats

# ============ Main loop ============
cv2.namedWindow('Depth Detection', cv2.WINDOW_NORMAL)
print("=" * 60)
print("🎬 Press 'Q' to quit | Press 'S' to screenshot")
print("=" * 60 + "\n")

fps_history = []
frame_count = 0
inference_times = []

try:
    while True:
        t_total_start = time.time()
        
        # Read frame
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue
        
        t_inference_start = time.time()
        
        # Depth inference (ensure running on CUDA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            if USE_CUDA:
                torch.cuda.synchronize()  # sync start
            
            depth = model.infer_image(rgb)
            
            if USE_CUDA:
                torch.cuda.synchronize()  # sync end, ensure accurate timing
        
        inference_time = time.time() - t_inference_start
        inference_times.append(inference_time)
        if len(inference_times) > 30:
            inference_times.pop(0)
        
        # Depth visualization
        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        
        # FPS calculation
        total_time = time.time() - t_total_start
        fps = 1.0 / total_time if total_time > 0 else 0
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)
        
        # System resources
        cpu = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory().percent
        gpu_stats = get_gpu_stats()
        
        # Info panel
        def add_info(img, title):
            h, w = img.shape[:2]
            
            # Semi-transparent background
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
            
            # Title
            cv2.putText(img, title, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Metrics
            y = 50
            # FPS
            cv2.putText(img, f"FPS: {avg_fps:.1f}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
            # Inference time
            avg_inference = np.mean(inference_times) * 1000
            cv2.putText(img, f"Inference: {avg_inference:.1f}ms", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 25
            # CPU
            cv2.putText(img, f"CPU: {cpu:.0f}%", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            y += 25
            # GPU
            gpu_text = f"GPU: {gpu_stats['load']:.0f}%"
            if USE_CUDA:
                gpu_text += f" ({gpu_stats['mem_used']:.1f}/{gpu_stats['mem_total']:.1f}GB)"
            cv2.putText(img, gpu_text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            y += 25
            # RAM
            cv2.putText(img, f"RAM: {mem:.0f}%", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            
            return img
        
        # Add info
        frame_vis = add_info(frame.copy(), "Original")
        depth_vis = add_info(depth_colored.copy(), "Depth Map")
        
        # Side-by-side display
        combined = np.hstack([frame_vis, depth_vis])
        
        # Show
        cv2.imshow('Depth Detection', combined)
        
        # Console output (every 30 frames)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[{frame_count:05d}] FPS: {avg_fps:.1f} | "
                  f"Inference: {np.mean(inference_times)*1000:.1f}ms | "
                  f"CPU: {cpu:.0f}% | GPU: {gpu_stats['load']:.0f}% | "
                  f"GPU Mem: {gpu_stats['mem_used']:.1f}GB")
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n👋 Exiting")
            break
        elif key == ord('s') or key == ord('S'):
            filename = f"depth_{frame_count:05d}.jpg"
            cv2.imwrite(filename, combined)
            print(f"📸 Saved: {filename}")

except KeyboardInterrupt:
    print("\n⚠️ Interrupted")

finally:
    cam.stop()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"✅ Test completed")
    print(f"   Total frames: {frame_count}")
    print(f"   Average FPS: {np.mean(fps_history):.1f}")
    print(f"   Average inference time: {np.mean(inference_times)*1000:.1f}ms")
    if USE_CUDA:
        print(f"   GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    print(f"{'='*60}")