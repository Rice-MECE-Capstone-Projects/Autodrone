"""
Depth Pro real-time depth estimation - Jetson Orin optimized (low-memory mode)
- Low memory usage (< 4GB GPU RAM)
- CUDA acceleration
- Real-time visualization
"""

import cv2
import torch
import numpy as np
import time
import psutil
from pathlib import Path
from threading import Thread
from queue import Queue
import sys
import gc
import os

# Add Depth Pro to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPTH_PRO_SRC = PROJECT_ROOT / 'depth_pro' / 'src'
sys.path.insert(0, str(DEPTH_PRO_SRC))

import depth_pro

# ============ Configuration ============
CAMERA_ID = 0
WIDTH, HEIGHT = 640, 480
USE_CUDA = torch.cuda.is_available()

# ✅ Unified model path
CHECKPOINT_PATH = PROJECT_ROOT / 'checkpoints' / 'depth_pro.pt'

print("🚀 Jetson Orin Depth Pro System")
print(f"   CUDA Available: {USE_CUDA}")
if USE_CUDA:
    print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   CUDA Memory: {mem_total:.1f} GB")
    
    torch.cuda.empty_cache()
    mem_free = torch.cuda.mem_get_info()[0] / 1e9
    print(f"   Free Memory: {mem_free:.1f} GB")
    
    if mem_free < 3.0:
        print("   ⚠️  Warning: available memory is less than 3GB, the model may fail to load")
        print("   💡 Tip: Close other programs or use CPU mode")

# ✅ Check model file
if not CHECKPOINT_PATH.exists():
    print(f"\n❌ Model file not found: {CHECKPOINT_PATH}")
    print("💡 Please run: bash setup_depth_pro.sh")
    sys.exit(1)

print(f"   Model path: {CHECKPOINT_PATH}")

# ============ Video capture thread ============
class VideoCapture:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.q = Queue(maxsize=2)
        self.stopped = False
        
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
print("📦 Loading Depth Pro model...")
print("   ⏳ This may take 30-60 seconds, please wait...")

try:
    if USE_CUDA:
        torch.cuda.empty_cache()
        gc.collect()
    
    device = torch.device("cuda" if USE_CUDA else "cpu")
    precision = torch.float32
    
    print(f"   Device: {device}")
    print(f"   Precision: {precision}")
    
    print("   📥 Loading model weights...")
    
    # ✅ Strategy: temporarily change working directory to checkpoint parent directory
    # so depth_pro can find ./checkpoints/depth_pro.pt
    original_cwd = os.getcwd()
    os.chdir(str(CHECKPOINT_PATH.parent.parent))  # switch to drone-obstacle-avoidance/
    
    try:
        # By default it will load from ./checkpoints/depth_pro.pt
        model, transform = depth_pro.create_model_and_transforms(
            device=device,
            precision=precision
        )
    finally:
        os.chdir(original_cwd)  # restore original directory
    
    if USE_CUDA:
        mem_used = torch.cuda.memory_allocated() / 1e9
        print(f"   ✅ Model loaded (GPU RAM: {mem_used:.2f} GB)")
    
    model.eval()
    
    if USE_CUDA:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print("   ✅ cuDNN optimization enabled")
        
        print("   🔍 Warming up model...")
        from PIL import Image
        dummy_pil = Image.new('RGB', (WIDTH, HEIGHT))
        dummy_tensor = transform(dummy_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _ = model.infer(dummy_tensor)
        torch.cuda.synchronize()
        
        mem_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"   ✅ Warmup done (Peak GPU RAM: {mem_peak:.2f} GB)")
    
    print("✅ Depth Pro loaded")

except RuntimeError as e:
    print(f"\n❌ Failed to load model: {e}")
    print("\n💡 Possible solutions:")
    print("   1. Close other programs that are using the GPU")
    print("   2. Reboot the system to free memory")
    print("   3. Use CPU mode (slower but stable):")
    print("      Change code USE_CUDA = False")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Load error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============ Initialize camera ============
print("📹 Starting camera...")
cam = VideoCapture(CAMERA_ID)
time.sleep(1)
print("✅ Camera ready\n")

# ============ GPU monitor ============
def get_gpu_stats():
    stats = {'load': 0, 'mem_used': 0, 'mem_total': 0}
    
    try:
        with open('/sys/devices/gpu.0/load', 'r') as f:
            stats['load'] = int(f.read().strip()) / 10
    except:
        pass
    
    if USE_CUDA:
        stats['mem_used'] = torch.cuda.memory_allocated() / 1e9
        stats['mem_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return stats

# ============ Main loop ============
cv2.namedWindow('Depth Pro', cv2.WINDOW_NORMAL)
print("=" * 60)
print("🎬 Press 'Q' to quit | Press 'S' to screenshot")
print("=" * 60 + "\n")

fps_history = []
frame_count = 0
inference_times = []

try:
    while True:
        t_total_start = time.time()
        
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue
        
        t_inference_start = time.time()
        
        from PIL import Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if USE_CUDA:
                torch.cuda.synchronize()
            
            prediction = model.infer(image_tensor)
            
            if USE_CUDA:
                torch.cuda.synchronize()
        
        depth = prediction["depth"].cpu().numpy()
        
        inference_time = time.time() - t_inference_start
        inference_times.append(inference_time)
        if len(inference_times) > 30:
            inference_times.pop(0)
        
        if depth.shape[:2] != (HEIGHT, WIDTH):
            depth_resized = cv2.resize(depth, (WIDTH, HEIGHT))
        else:
            depth_resized = depth
        
        depth_norm = ((depth_resized - depth_resized.min()) / 
                      (depth_resized.max() - depth_resized.min()) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        
        total_time = time.time() - t_total_start
        fps = 1.0 / total_time if total_time > 0 else 0
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)
        
        cpu = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory().percent
        gpu_stats = get_gpu_stats()
        
        def add_info(img, title):
            h, w = img.shape[:2]
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
            
            cv2.putText(img, title, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y = 50
            cv2.putText(img, f"FPS: {avg_fps:.1f}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
            avg_inference = np.mean(inference_times) * 1000
            cv2.putText(img, f"Inference: {avg_inference:.1f}ms", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 25
            cv2.putText(img, f"CPU: {cpu:.0f}%", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            y += 25
            gpu_text = f"GPU: {gpu_stats['load']:.0f}%"
            if USE_CUDA:
                gpu_text += f" ({gpu_stats['mem_used']:.1f}/{gpu_stats['mem_total']:.1f}GB)"
            cv2.putText(img, gpu_text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            y += 25
            cv2.putText(img, f"RAM: {mem:.0f}%", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            
            return img
        
        frame_vis = add_info(frame.copy(), "Original")
        depth_vis = add_info(depth_colored.copy(), "Depth Pro")
        combined = np.hstack([frame_vis, depth_vis])
        
        cv2.imshow('Depth Pro', combined)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[{frame_count:05d}] FPS: {avg_fps:.1f} | "
                  f"Inference: {np.mean(inference_times)*1000:.1f}ms | "
                  f"CPU: {cpu:.0f}% | GPU: {gpu_stats['load']:.0f}% | "
                  f"GPU Mem: {gpu_stats['mem_used']:.1f}GB")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n👋 Exiting")
            break
        elif key == ord('s') or key == ord('S'):
            filename = f"depth_pro_{frame_count:05d}.jpg"
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
    if fps_history:
        print(f"   Average FPS: {np.mean(fps_history):.1f}")
    if inference_times:
        print(f"   Average inference time: {np.mean(inference_times)*1000:.1f}ms")
    if USE_CUDA:
        print(f"   GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    print(f"{'='*60}")