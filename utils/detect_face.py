"""
Utility functions for face detection using YOLO model.
"""

from ultralytics import YOLO
import cv2
import os

# Load the YOLO model for face detection
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_PATH = os.path.join(BASE_DIR, 'Weight', 'yolov8n-face.pt')

print("YOLO weight path:", WEIGHT_PATH)
print("Weight exists:", os.path.exists(WEIGHT_PATH))

yolo_model = YOLO(WEIGHT_PATH)


def detect_face(frame):
    """
    Detect faces in the frame using YOLO model.

    Args:
        frame (numpy.ndarray): Input frame.

    Returns:
        list: List of bounding boxes [x1, y1, x2, y2].
    """
    # Perform face detection with YOLO
    result = yolo_model.predict(frame, conf=0.4, iou=0.5)

    boxes = result[0].boxes
    bounding_boxes = []

    # Extract bounding box coordinates
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].xyxy.cpu().numpy()[0][:4]
        bounding_boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return bounding_boxes


