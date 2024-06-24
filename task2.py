import subprocess
import sys

# Function to install a package
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--user"])

# Ensure necessary packages are installed
try:
    import cv2
except ImportError:
    install('opencv-contrib-python')

try:
    import torch
except ImportError:
    install('torch')

try:
    import ultralytics
except ImportError:
    install('ultralytics')

import cv2
import torch

# Function to extract frames from video
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/frame_{frame_count:04d}.jpg", frame)
        frame_count += 1
    cap.release()

# Load YOLOv5 model (use your trained model path if you have one)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to draw bounding boxes
def draw_boxes(frame, results):
    for box in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Initialize video capture for real-time detection
cap = cv2.VideoCapture(0)

# Tracker setup (using CSRT tracker)
tracker = cv2.TrackerCSRT_create()
initialized = False
bbox = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if not initialized:
        results = model(frame)
        draw_boxes(frame, results)
        
        # Initialize the tracker with the first detected object
        if len(results.xyxy[0]) > 0:
            bbox = (int(results.xyxy[0][0][0]), int(results.xyxy[0][0][1]), 
                    int(results.xyxy[0][0][2] - results.xyxy[0][0][0]), 
                    int(results.xyxy[0][0][3] - results.xyxy[0][0][1]))
            tracker.init(frame, bbox)
            initialized = True
    else:
        # Update the tracker and draw the tracked bounding box
        success, bbox = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            initialized = False

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

