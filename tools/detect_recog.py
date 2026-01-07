"""
Main handler for face detection and recognition.
"""

import cv2
from utils.detect_face import detect_face
from utils.embed import extract_feature
from models.facenet import face_net


def normalize(img):
    """
    Normalize image by subtracting mean and dividing by std.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Normalized image.
    """
    mean, std = img.mean(), img.std()
    return (img - mean) / std


def detect_and_recogface(frame, l2_normalize, embedding_df):
    """
    Detect and recognize faces in the frame.

    Args:
        frame (numpy.ndarray): Input frame from webcam.
        l2_normalize (Normalizer): L2 normalizer.
        embedding_df (pd.DataFrame): DataFrame with embeddings and labels.

    Returns:
        numpy.ndarray: Frame with drawn rectangles and labels.
    """
    # Detect faces in the frame
    bounding_boxes = detect_face(frame)
    
    # Process each detected face
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        
        # Draw rectangle around the face
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 225), 4)

        # Crop and preprocess the face image
        img_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        img_crop = normalize(img_crop)
        img_crop = cv2.resize(img_crop, (160, 160))

        # Extract embedding vector
        vector_embedding = extract_feature(img_crop)
        vector_embedding = l2_normalize.transform(vector_embedding.reshape(1, -1))[0]

        # Recognize the person
        person_name = face_net(vector_embedding, embedding_df)
        
        # Put text label on the frame
        cv2.putText(frame, person_name, (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

