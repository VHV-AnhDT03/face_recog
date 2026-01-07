import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import pandas as pd
import cv2
import platform

from tools.detect_recog import detect_and_recogface
from sklearn.preprocessing import Normalizer

# ===============================
# Load embedding database
# ===============================
embedding_df = pd.read_csv('./embedding_with_label.csv')

# Initialize L2 normalizer
l2_normalize = Normalizer('l2')

# ===============================
# Initialize webcam (Windows / macOS)
# ===============================
def init_camera():
    system = platform.system()

    if system == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:  # macOS / Linux
        cap = cv2.VideoCapture(0)

    # Force resolution (ổn định hơn cho YOLO)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Can't open webcam")
        sys.exit(1)

    return cap


cap = init_camera()
print("✅ Webcam started")

# ===============================
# Main loop
# ===============================
while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("❌ Can't receive frame from webcam")
        break

    # ---- Face detect & recognize (giữ nguyên luồng của bạn) ----
    frame = detect_and_recogface(frame, l2_normalize, embedding_df)

    # Display result
    cv2.imshow('Webcam Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# Cleanup
# ===============================
cap.release()
cv2.destroyAllWindows()