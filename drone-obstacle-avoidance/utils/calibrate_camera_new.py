"""
Camera Calibration Tool - Image Folder Version
Read pre-captured images from a folder and perform calibration

Usage:
1. Capture 15-20 images with different angles using your phone
2. Place all images in a folder (e.g., calibration_images/)
3. Run: python3 calibrate_camera_new.py --images calibration_images/
4. Output: camera_calib.npz + calibration_report.txt
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

# ============ Configuration ============
CHESSBOARD_SIZE = (6, 5)  # Inner corners (columns x rows)
SQUARE_SIZE = 0.030        # Square size in meters (measure with ruler!)

def main():
    parser = argparse.ArgumentParser(description="Camera Calibration from Image Folder")
    parser.add_argument("--images", type=str, required=True,
                       help="Path to folder containing calibration images")
    parser.add_argument("--square-size", type=float, default=SQUARE_SIZE,
                       help="Chessboard square size in meters (default: 0.030)")
    parser.add_argument("--output", type=str, default="camera_calib.npz",
                       help="Output calibration file name")
    args = parser.parse_args()
    
    img_folder = Path(args.images)
    if not img_folder.exists():
        print(f"❌ Folder not found: {img_folder}")
        return
    
    # Find all image files
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = []
    for ext in img_extensions:
        img_files.extend(img_folder.glob(f'*{ext}'))
        img_files.extend(img_folder.glob(f'*{ext.upper()}'))
    
    img_files = sorted(img_files)
    
    if len(img_files) == 0:
        print(f"❌ No images found in {img_folder}")
        return
    
    print("=" * 60)
    print("📷 Camera Calibration - Image Folder Mode")
    print("=" * 60)
    print(f"   Chessboard: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners")
    print(f"   Square size: {args.square_size*1000:.1f}mm")
    print(f"   Found {len(img_files)} images in {img_folder}\n")
    
    # Prepare object points (3D world coordinates, Z=0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= args.square_size
    
    # Storage for valid calibration data
    objpoints = []  # 3D world points
    imgpoints = []  # 2D image points
    
    # Corner refinement criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    valid_count = 0
    img_width, img_height = None, None
    
    print("🔍 Processing images...\n")
    
    for i, img_path in enumerate(img_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  [{i}/{len(img_files)}] {img_path.name}: Failed to read")
            continue
        
        # Store image dimensions from first valid image
        if img_width is None:
            img_height, img_width = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Refine corner locations
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            
            valid_count += 1
            print(f"✅ [{i}/{len(img_files)}] {img_path.name}: Pattern detected")
            
            # Optional: save annotated image for verification
            img_annotated = img.copy()
            cv2.drawChessboardCorners(img_annotated, CHESSBOARD_SIZE, corners_refined, ret)
            output_folder = img_folder / "annotated"
            output_folder.mkdir(exist_ok=True)
            cv2.imwrite(str(output_folder / img_path.name), img_annotated)
        else:
            print(f"❌ [{i}/{len(img_files)}] {img_path.name}: No pattern detected")
    
    print(f"\n📊 Summary: {valid_count}/{len(img_files)} images valid\n")
    
    if valid_count < 10:
        print("❌ Insufficient valid images (need at least 10)")
        print("   Recommendations:")
        print("   - Check image quality (focus, lighting)")
        print("   - Ensure entire chessboard is visible")
        print("   - Verify SQUARE_SIZE matches your printed board")
        return
    
    # ============ Run Calibration ============
    print("📐 Running cv2.calibrateCamera()...\n")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (img_width, img_height), None, None
    )
    
    if not ret:
        print("❌ Calibration failed")
        return
    
    print("✅ Calibration successful!\n")
    
    # ============ Print Results ============
    print("=" * 60)
    print("📊 Calibration Results")
    print("=" * 60)
    
    print("\n📷 Camera Matrix (K):")
    print(camera_matrix)
    
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    print(f"\n   fx = {fx:.2f} pixels")
    print(f"   fy = {fy:.2f} pixels")
    print(f"   cx = {cx:.2f} pixels")
    print(f"   cy = {cy:.2f} pixels")
    
    print("\n🔍 Distortion Coefficients:")
    print(dist_coeffs.ravel())
    
    k1, k2, p1, p2, k3 = dist_coeffs.ravel()
    print(f"\n   k1 (radial)     = {k1:.6f}")
    print(f"   k2 (radial)     = {k2:.6f}")
    print(f"   p1 (tangential) = {p1:.6f}")
    print(f"   p2 (tangential) = {p2:.6f}")
    print(f"   k3 (radial)     = {k3:.6f}")
    
    # ============ Reprojection Error ============
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                          camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    mean_error /= len(objpoints)
    
    print(f"\n📏 Re-projection Error: {mean_error:.4f} pixels")
    
    if mean_error < 0.5:
        print("   ✅ Excellent! (<0.5 px)")
        quality = "Excellent"
    elif mean_error < 1.0:
        print("   ✅ Good! (<1.0 px)")
        quality = "Good"
    else:
        print("   ⚠️  High error. Consider adding more images or improving quality.")
        quality = "Fair"
    
    # ============ Save Results ============
    # 🔧 FIX: Save to script directory, not working directory
    script_dir = Path(__file__).parent
    output_path = script_dir / args.output
    
    np.savez(str(output_path),
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             img_width=img_width,
             img_height=img_height,
             reprojection_error=mean_error,
             num_images=valid_count)
    
    print(f"\n💾 Calibration saved to: {output_path}")
    
    # ============ Generate Test Undistortion ============
    print("\n🔬 Generating undistortion test...")
    
    # Use first valid image
    test_img_path = None
    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, _ = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            if ret:
                test_img_path = img_path
                break
    
    if test_img_path:
        test_img = cv2.imread(str(test_img_path))
        h, w = test_img.shape[:2]
        
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        # Undistort
        dst = cv2.undistort(test_img, camera_matrix, dist_coeffs, None, newcameramtx)
        
        # Crop ROI
        x, y, w_roi, h_roi = roi
        dst_cropped = dst[y:y+h_roi, x:x+w_roi]
        
        # Create comparison
        comparison = np.hstack([
            cv2.resize(test_img, (w_roi, h_roi)),
            dst_cropped
        ])
        
        # 🔧 FIX: Save to script directory
        test_output = script_dir / 'calibration_test.jpg'
        cv2.imwrite(str(test_output), comparison)
        print(f"   💾 Test image saved: {test_output}")
        print("   (Left: original | Right: undistorted)")
    
    # ============ Generate Report ============
    report_path = output_path.with_suffix('.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Camera Calibration Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Script directory: {script_dir}\n")
        f.write(f"Images folder: {img_folder}\n")
        f.write(f"Valid images: {valid_count}/{len(img_files)}\n")
        f.write(f"Image resolution: {img_width}x{img_height}\n")
        f.write(f"Chessboard: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners\n")
        f.write(f"Square size: {args.square_size*1000:.1f}mm\n\n")
        
        f.write("Camera Matrix (K):\n")
        f.write(str(camera_matrix) + "\n\n")
        f.write(f"fx = {fx:.2f} pixels\n")
        f.write(f"fy = {fy:.2f} pixels\n")
        f.write(f"cx = {cx:.2f} pixels\n")
        f.write(f"cy = {cy:.2f} pixels\n\n")
        
        f.write("Distortion Coefficients:\n")
        f.write(str(dist_coeffs.ravel()) + "\n\n")
        f.write(f"k1 = {k1:.6f}\n")
        f.write(f"k2 = {k2:.6f}\n")
        f.write(f"p1 = {p1:.6f}\n")
        f.write(f"p2 = {p2:.6f}\n")
        f.write(f"k3 = {k3:.6f}\n\n")
        
        f.write(f"Re-projection Error: {mean_error:.4f} pixels ({quality})\n")
    
    print(f"   💾 Report saved: {report_path}")
    
    print("\n" + "=" * 60)
    print("✅ Calibration complete!")
    print("   Next steps:")
    print(f"   1. Review {test_output}")
    print(f"   2. Check {report_path} for full details")
    print("   3. Run avoidance_midas_pro.py with calibration")
    print("=" * 60)

if __name__ == "__main__":
    main()