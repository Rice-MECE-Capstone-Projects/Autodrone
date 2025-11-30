import cv2
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python3 diagnose_image.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
img = cv2.imread(img_path)

if img is None:
    print(f"❌ Failed to read image: {img_path}")
    sys.exit(1)

print(f"✅ Image loaded successfully")
print(f"   Resolution: {img.shape[1]}x{img.shape[0]}")
print(f"   Channels: {img.shape[2]}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Try multiple chessboard sizes
sizes_to_try = [
    (6, 5, "6x5 (your config)"),  # 🔧 First item changed to (6, 5)
    (5, 6, "5x6 (swapped)"),      # 🔧 Second item changed to (5, 6)
    (7, 6, "7x6"),
    (6, 7, "6x7"),
    (8, 6, "8x6"),
    (6, 8, "6x8"),
    (9, 6, "9x6"),
    (6, 9, "6x9"),
]

print("\n🔍 Testing different chessboard sizes:")
for cols, rows, desc in sizes_to_try:
    ret, corners = cv2.findChessboardCorners(
        gray, (cols, rows),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    status = "✅ FOUND" if ret else "❌ Not found"
    print(f"   {desc:20s} {status}")
    
    if ret:
        print(f"\n🎉 SUCCESS! Detected {cols}x{rows} pattern")
        print(f"   You should use: CHESSBOARD_SIZE = ({cols}, {rows})")
        
        # Save annotated version
        img_annotated = img.copy()
        cv2.drawChessboardCorners(img_annotated, (cols, rows), corners, ret)
        output_path = f"diagnostic_{Path(img_path).stem}_detected.jpg"
        cv2.imwrite(output_path, img_annotated)
        print(f"   💾 Saved annotated image: {output_path}")
        break
else:
    print("\n⚠️  No standard pattern detected")
    print("   Possible issues:")
    print("   - Image is rotated/flipped")
    print("   - Chessboard is not fully visible")
    print("   - Image is too blurry")
    print("   - Lighting/contrast is poor")
    
    # Save grayscale for manual inspection
    cv2.imwrite(f"diagnostic_{Path(img_path).stem}_gray.jpg", gray)
    print(f"   💾 Saved grayscale: diagnostic_{Path(img_path).stem}_gray.jpg")