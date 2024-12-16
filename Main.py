import math
import time
import cv2
import cvzone
from ultralytics import YOLO

# Minimum confidence threshold for detections
confidence = 0.6

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)  # Use '0' for default webcam, '1' if you have multiple cameras
cap.set(3, 640)  # Set frame width
cap.set(4, 480)  # Set frame height

# Load YOLO model from the specified path
model = YOLO("Face Anti-Spoofing(mini project)\\best.pt")

# Class names for the detections
classNames = ["fake", "real"]

# Variables to calculate FPS
prev_frame_time = 0
new_frame_time = 0

# Main loop for processing video frames
while True:
    # Update time for current frame
    new_frame_time = time.time()
    
    # Capture frame from webcam
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    
    # Run YOLO model on the captured frame
    results = model(img, stream=True, verbose=False)
    
    # Process each detection
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            # Confidence of the detection
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Class of the detection
            cls = int(box.cls[0])
            
            # Only process detections above the confidence threshold
            if conf > confidence:
                if classNames[cls] == 'real':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                # Draw bounding box and label
                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color, colorB=color)

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps:.2f}")

    # Display the frame with detections
    cv2.imshow("Image", img)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()