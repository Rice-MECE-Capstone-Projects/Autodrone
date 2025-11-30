import cv2
from pathlib import Path

img_folder = Path("calibration_images")
img_files = sorted(img_folder.glob("*.jpg"))

print("=" * 80)
print(f"Checking {len(img_files)} images...")
print("=" * 80)

for i, img_path in enumerate(img_files, 1):
    img = cv2.imread(str(img_path))
    
    if img is None:
        print(f"❌ [{i:2d}] {img_path.name:30s} FAILED TO READ")
        continue
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Quick check for 6x5 (modify here)
    ret_65, _ = cv2.findChessboardCorners(gray, (6, 5), None)  # 🔧 Change to (6, 5)
    # Also try 5x6 (swapped)
    ret_56, _ = cv2.findChessboardCorners(gray, (5, 6), None)  # 🔧 Change to (5, 6)
    
    status = ""
    if ret_65:  # 🔧 Change to ret_65
        status = "✅ 6x5"  # 🔧 Change to 6x5
    elif ret_56:  # 🔧 Change to ret_56
        status = "⚠️  5x6 (swapped!)"  # 🔧 Change to 5x6
    else:
        status = "❌ NOT DETECTED"
    
    print(f"[{i:2d}] {img_path.name:30s} {w:4d}x{h:4d}  {status}")

print("=" * 80)