"""
Camera Calibration Tool for Drone Avoidance System
Use a chessboard to calibrate camera intrinsics and distortion coefficients

Usage:
1. Print the chessboard pattern (7x6 inner corners, 30mm squares)
2. Fix the chessboard on a wall or flat surface
3. Run the script and move the camera to different positions/angles
4. Press SPACE to capture 15–20 images
5. Press Q to finish calibration
"""

import cv2
import numpy as np
import time
from pathlib import Path

# ============ Configuration ============
CHESSBOARD_SIZE = (6, 5)  # Number of inner corners (cols × rows)
SQUARE_SIZE = 0.030        # Chessboard square side length (meters), please measure!
CAMERA_ID = 0
REQUIRED_IMAGES = 15       # Recommended number of images

# ============ Initialization ============
print("=" * 60)
print("📷 Camera Calibration Tool")
print("=" * 60)
print(f"   Chessboard: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners")
print(f"   Square size: {SQUARE_SIZE*1000:.1f}mm")
print(f"   Recommended images: {REQUIRED_IMAGES}+")
print("\n📋 Instructions:")
print("   1. Fix the printed chessboard on a wall/flat surface")
print("   2. Move the camera to different positions:")
print("      - Close (30cm), medium (50cm), far (70cm)")
print("      - Left/right tilt (±30°)")
print("      - Up/down tilt (±30°)")
print("      - Four corners of the frame")
print("   3. Press SPACE when pattern is detected and stable")
print("   4. Press Q when you have 15+ good images\n")

# Create capture directory
capture_dir = Path(__file__).parent / 'calibration_captures'
capture_dir.mkdir(exist_ok=True)
print(f"💾 Captured images will be saved to: {capture_dir}\n")

# Prepare chessboard 3D points (Z=0 plane)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Store corner points of all valid images
objpoints = []  # 3D world coordinates
imgpoints = []  # 2D image coordinates
captured_frames = []  # Store original images for later inspection

# Initialize camera
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Failed to open camera")
    exit(1)

# Get actual resolution
ret, test_frame = cap.read()
if ret:
    img_height, img_width = test_frame.shape[:2]
    print(f"✅ Camera initialized: {img_width}x{img_height}\n")
else:
    print("❌ Failed to read test frame")
    exit(1)

# Corner refinement criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

captured_count = 0
last_capture_time = 0
MIN_CAPTURE_INTERVAL = 1.0  # Minimum capture interval (seconds)

# Capture suggestions (to guide user step by step)
capture_suggestions = [
    "Front view, medium distance (50cm)",
    "Front view, close (30cm)",
    "Front view, far (70cm)",
    "Left tilt 30°, medium distance",
    "Right tilt 30°, medium distance",
    "Up tilt 30°, medium distance",
    "Down tilt 30°, medium distance",
    "Top-left corner of frame",
    "Top-right corner of frame",
    "Bottom-left corner of frame",
    "Bottom-right corner of frame",
    "Rotate camera 45°, center",
    "Free angle 1",
    "Free angle 2",
    "Free angle 3",
]

def get_blur_score(gray):
    """Compute image sharpness (Laplacian variance)"""
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def check_duplicate(new_corners, imgpoints, threshold=10.0):
    """Check whether the new corners are too similar to existing ones"""
    if len(imgpoints) == 0:
        return False
    
    for existing in imgpoints:
        diff = np.abs(new_corners - existing).mean()
        if diff < threshold:
            return True
    return False

try:
    print("🎬 Starting live preview...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret_corners, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        # Display frame
        display = frame.copy()
        
        # Current suggestion
        if captured_count < len(capture_suggestions):
            suggestion = capture_suggestions[captured_count]
        else:
            suggestion = f"Extra image {captured_count - len(capture_suggestions) + 1}"
        
        # Top info bar
        info_bg = np.zeros((120, display.shape[1], 3), dtype=np.uint8)
        
        if ret_corners:
            # Compute sharpness
            blur_score = get_blur_score(gray)
            
            # Check duplicate
            is_duplicate = check_duplicate(corners, imgpoints)
            
            # Draw detected corners
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, ret_corners)
            
            # Status text
            if blur_score < 100:
                status = "⚠️  Too blurry! Hold still"
                color = (0, 165, 255)  # Orange
            elif is_duplicate:
                status = "⚠️  Too similar! Change angle"
                color = (0, 165, 255)
            else:
                status = "✅ Pattern OK - Press SPACE"
                color = (0, 255, 0)  # Green
            
            cv2.putText(info_bg, status, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(info_bg, f"Blur score: {blur_score:.1f} (need >100)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(info_bg, "❌ No pattern detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(info_bg, "Move camera until chessboard is fully visible", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Progress and suggestion
        cv2.putText(info_bg, f"Progress: {captured_count}/{REQUIRED_IMAGES}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(info_bg, f"Next: {suggestion}", 
                   (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Combine info bar and frame
        combined = np.vstack([info_bg, display])
        
        cv2.imshow('Camera Calibration', combined)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Press SPACE to capture
        if key == ord(' ') and ret_corners:
            current_time = time.time()
            
            if current_time - last_capture_time < MIN_CAPTURE_INTERVAL:
                print(f"⚠️  Wait {MIN_CAPTURE_INTERVAL:.1f}s between captures")
                continue
            
            # Check sharpness
            blur_score = get_blur_score(gray)
            if blur_score < 100:
                print(f"⚠️  Image too blurry (score: {blur_score:.1f}), try again")
                continue
            
            # Refine corners
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Check duplicate
            if check_duplicate(corners_refined, imgpoints):
                print(f"⚠️  Too similar to existing image, change angle")
                continue
            
            # Save points
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            captured_frames.append(frame.copy())
            
            # Save image to disk
            img_filename = capture_dir / f"capture_{captured_count:02d}.jpg"
            cv2.imwrite(str(img_filename), frame)
            
            captured_count += 1
            last_capture_time = current_time
            
            print(f"✅ Image {captured_count} captured (blur: {blur_score:.1f}) -> {img_filename.name}")
            
            # Flash effect
            flash = np.ones_like(combined) * 255
            cv2.imshow('Camera Calibration', flash)
            cv2.waitKey(100)
        
        # Press Q to quit
        elif key == ord('q') or key == ord('Q'):
            if captured_count < 10:
                print(f"\n⚠️  Warning: Only {captured_count} images captured (recommended: 15+)")
                print("   Continue anyway? (y/n): ", end='')
                import sys
                sys.stdout.flush()
                response = input().strip().lower()
                if response != 'y':
                    print("❌ Calibration cancelled")
                    break
            
            if captured_count == 0:
                print("❌ No images captured")
                break
            
            print(f"\n🔧 Starting calibration with {captured_count} images...")
            break
        
        # Press R to reset (remove last one)
        elif key == ord('r') or key == ord('R'):
            if captured_count > 0:
                objpoints.pop()
                imgpoints.pop()
                captured_frames.pop()
                captured_count -= 1
                print(f"🔄 Last image removed, now: {captured_count} images")

except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()

# ============ Run calibration ============
if captured_count > 0:
    print("\n📐 Running cv2.calibrateCamera()...")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (img_width, img_height), None, None
    )
    
    if not ret:
        print("❌ Calibration failed")
        exit(1)
    
    print("✅ Calibration successful!\n")
    
    # ============ Print results ============
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
    
    # ============ Re-projection error ============
    mean_error = 0
    errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                          camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors.append(error)
        mean_error += error
    
    mean_error /= len(objpoints)
    
    print(f"\n📏 Re-projection Error: {mean_error:.4f} pixels")
    print(f"   Min error: {min(errors):.4f} px")
    print(f"   Max error: {max(errors):.4f} px")
    
    if mean_error < 0.5:
        print("   ✅ Excellent! (<0.5 px)")
        quality = "Excellent"
    elif mean_error < 1.0:
        print("   ✅ Good! (<1.0 px)")
        quality = "Good"
    else:
        print("   ⚠️  High error. Consider recapturing with better images.")
        quality = "Fair"
    
    # ============ Save results ============
    output_path = Path(__file__).parent / 'camera_calib.npz'
    
    np.savez(str(output_path),
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             img_width=img_width,
             img_height=img_height,
             reprojection_error=mean_error,
             num_images=captured_count)
    
    print(f"\n💾 Calibration saved to: {output_path}")
    
    # ============ Generate annotated captured images ============
    print("\n🖼️  Generating annotated images...")
    annotated_dir = capture_dir / 'annotated'
    annotated_dir.mkdir(exist_ok=True)
    
    for i, (frame, corners) in enumerate(zip(captured_frames, imgpoints)):
        annotated = frame.copy()
        cv2.drawChessboardCorners(annotated, CHESSBOARD_SIZE, corners, True)
        
        # Add error info
        error_text = f"Error: {errors[i]:.4f}px"
        cv2.putText(annotated, error_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(str(annotated_dir / f"annotated_{i:02d}.jpg"), annotated)
    
    print(f"   💾 Saved to: {annotated_dir}/")
    
    # ============ Generate undistortion test image ============
    print("\n🔬 Generating undistortion test...")
    
    # Use the first captured image
    test_img = captured_frames[0]
    h, w = test_img.shape[:2]
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    # Undistort
    dst = cv2.undistort(test_img, camera_matrix, dist_coeffs, None, newcameramtx)
    
    # Crop ROI
    x, y, w_roi, h_roi = roi
    dst_cropped = dst[y:y+h_roi, x:x+w_roi]
    
    # Save comparison image
    comparison = np.hstack([
        cv2.resize(test_img, (w_roi, h_roi)),
        dst_cropped
    ])
    
    test_path = Path(__file__).parent / 'calibration_test.jpg'
    cv2.imwrite(str(test_path), comparison)
    print(f"   💾 Test image saved: {test_path}")
    print("   (Left: original | Right: undistorted)")
    
    # ============ Generate detailed report ============
    report_path = Path(__file__).parent / 'camera_calib.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Camera Calibration Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Captured images: {captured_count}\n")
        f.write(f"Image resolution: {img_width}x{img_height}\n")
        f.write(f"Chessboard: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners\n")
        f.write(f"Square size: {SQUARE_SIZE*1000:.1f}mm\n\n")
        
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
        f.write(f"Min error: {min(errors):.4f} px\n")
        f.write(f"Max error: {max(errors):.4f} px\n\n")
        
        f.write("Per-image errors:\n")
        for i, err in enumerate(errors):
            f.write(f"  Image {i:2d}: {err:.4f} px\n")
    
    print(f"   💾 Report saved: {report_path}")
    
    print("\n" + "=" * 60)
    print("✅ Calibration complete!")
    print("\n📂 Generated files:")
    print(f"   - {output_path} (calibration data)")
    print(f"   - {test_path} (undistortion test)")
    print(f"   - {report_path} (detailed report)")
    print(f"   - {capture_dir}/ (original images)")
    print(f"   - {annotated_dir}/ (annotated images)")
    print("\n📋 Next steps:")
    print("   1. Review calibration_test.jpg for quality")
    print("   2. Check camera_calib.txt for details")
    print("   3. Run: python3 avoidance_midas_pro.py --method size_priors")
    print("=" * 60)

else:
    print("\n❌ No calibration performed")