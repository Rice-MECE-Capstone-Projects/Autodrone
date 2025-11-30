"""Quick verification of chessboard detection with correct size"""
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

CHESSBOARD_SIZE = (6, 5)  # Correct size

print(f"🔍 Testing chessboard detection: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners")
print("   (7 columns × 6 rows of squares)")
print("\nPress Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    
    if ret_corners:
        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, True)
        cv2.putText(frame, f"✅ DETECTED {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        print("✅ Pattern detected!")
    else:
        cv2.putText(frame, "❌ No pattern", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    cv2.imshow('Verification', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()