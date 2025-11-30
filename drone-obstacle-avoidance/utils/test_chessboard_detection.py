"""
Chessboard Detection Diagnostic Tool
Test different chessboard configurations and detection parameters
"""

import cv2
import numpy as np
import time
from pathlib import Path

print("=" * 60)
print("🔍 Chessboard Detection Diagnostic")
print("=" * 60)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Failed to open camera")
    exit(1)

print("✅ Camera opened")
print("\n📋 Controls:")
print("   S - Save current frame for analysis")
print("   T - Toggle detection method")
print("   1-9 - Try different chessboard sizes")
print("   Q - Quit\n")

# Chessboard configurations to test
chessboard_configs = [
    ((6, 5), "6x5 (your config)"),  # 🔧 First item changed to (6, 5)
    ((5, 6), "5x6 (swapped)"),      # 🔧 Second item changed to (5, 6)
    ((7, 6), "7x6"),
    ((6, 7), "6x7"),
    ((8, 6), "8x6"),
    ((6, 8), "6x8"),
    ((9, 6), "9x6"),
    ((6, 9), "6x9"),
    ((7, 5), "7x5"),
]

current_config_idx = 0
use_adaptive = True

def test_detection(gray, size, method_name="Standard"):
    """Test different detection methods"""
    results = []
    
    # Method 1: Standard detection
    flags1 = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret1, corners1 = cv2.findChessboardCorners(gray, size, flags1)
    results.append(("Standard", ret1, corners1))
    
    # Method 2: Without FastCheck
    flags2 = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret2, corners2 = cv2.findChessboardCorners(gray, size, flags2)
    results.append(("No FastCheck", ret2, corners2))
    
    # Method 3: Simple detection
    ret3, corners3 = cv2.findChessboardCorners(gray, size, None)
    results.append(("Simple", ret3, corners3))
    
    # Method 4: Detect after contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    ret4, corners4 = cv2.findChessboardCorners(enhanced, size, flags1)
    results.append(("Enhanced", ret4, corners4))
    
    return results

print("🎬 Starting live detection test...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display = frame.copy()
    
    # Current configuration
    size, desc = chessboard_configs[current_config_idx]
    
    # Test all methods
    detection_results = test_detection(gray, size)
    
    # Find best result
    best_result = None
    for method, ret_corners, corners in detection_results:
        if ret_corners:
            best_result = (method, corners)
            break
    
    # Info bar
    info_height = 200
    info_bg = np.zeros((info_height, display.shape[1], 3), dtype=np.uint8)
    
    y_pos = 25
    cv2.putText(info_bg, f"Testing: {desc}", 
               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_pos += 30
    cv2.putText(info_bg, f"Image: {gray.shape[1]}x{gray.shape[0]}, Mean brightness: {gray.mean():.1f}", 
               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    y_pos += 35
    cv2.putText(info_bg, "Detection Methods:", 
               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1)
    
    # Show each method result
    for method, ret_corners, corners in detection_results:
        y_pos += 25
        status = "✅" if ret_corners else "❌"
        color = (0, 255, 0) if ret_corners else (0, 0, 255)
        cv2.putText(info_bg, f"{status} {method:15s}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw corners if detected
    if best_result:
        method, corners = best_result
        cv2.drawChessboardCorners(display, size, corners, True)
        
        # Success text
        success_text = f"DETECTED with {method}!"
        cv2.putText(display, success_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    # Combine display
    combined = np.vstack([info_bg, display])
    cv2.imshow('Chessboard Detection Test', combined)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        break
    
    elif key == ord('s') or key == ord('S'):
        # Save current frame for analysis
        save_dir = Path(__file__).parent / 'debug_frames'
        save_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        
        # Save original image
        cv2.imwrite(str(save_dir / f"frame_{timestamp}_original.jpg"), frame)
        
        # Save grayscale image
        cv2.imwrite(str(save_dir / f"frame_{timestamp}_gray.jpg"), gray)
        
        # Save contrast-enhanced image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        cv2.imwrite(str(save_dir / f"frame_{timestamp}_enhanced.jpg"), enhanced)
        
        # Save detection report
        with open(save_dir / f"frame_{timestamp}_report.txt", 'w') as f:
            f.write(f"Chessboard size tested: {size}\n")
            f.write(f"Image resolution: {gray.shape[1]}x{gray.shape[0]}\n")
            f.write(f"Mean brightness: {gray.mean():.1f}\n\n")
            f.write("Detection results:\n")
            for method, ret_corners, _ in detection_results:
                f.write(f"  {method:15s}: {'SUCCESS' if ret_corners else 'FAILED'}\n")
        
        print(f"💾 Saved debug frames to: {save_dir}/")
    
    elif ord('1') <= key <= ord('9'):
        idx = key - ord('1')
        if idx < len(chessboard_configs):
            current_config_idx = idx
            size, desc = chessboard_configs[current_config_idx]
            print(f"🔄 Switched to: {desc}")

cap.release()
cv2.destroyAllWindows()

print("\n✅ Diagnostic complete")
print("\n💡 Troubleshooting tips:")
print("   1. Check if any detection method succeeded")
print("   2. Try different chessboard sizes (1-9 keys)")
print("   3. Ensure the entire chessboard is visible in frame")
print("   4. Improve lighting (avoid shadows and reflections)")
print("   5. Make sure the chessboard is flat and not wrinkled")
print("   6. If saved frames, check debug_frames/ folder")