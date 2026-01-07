"""
Utility functions for face embedding extraction and processing.
"""

import os
import cv2
import numpy as np
import pandas as pd
import csv

from tensorflow.keras.models import load_model
from models.inception_resnet import InceptionResNetV2
from utils.detect_face import detect_face

# Initialize and load the Facenet model with pre-trained weights
facenet_model = InceptionResNetV2()
weight_path = './Weight/facenet_keras_weights.h5'
facenet_model.load_weights(weight_path)


def normalize(image):
    """
    Normalize image by dividing by 255.0.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Normalized image.
    """
    return image / 255.0


def extract_feature(image):
    """
    Extract feature vector from image using Facenet model.

    Args:
        image (numpy.ndarray): Input image of shape (160, 160, 3).

    Returns:
        numpy.ndarray: Feature vector.
    """
    image = np.expand_dims(image, axis=0)
    return facenet_model.predict(image)


def create_embedding(folder_path):
    """
    Create embeddings from images in folder.

    Args:
        folder_path (str): Path to folder containing subfolders of images.

    Returns:
        tuple: (embeddings array, labels list)
    """
    embeddings = []
    labels = []

    # Iterate through each label (person) folder
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)

        if os.path.isdir(label_folder):
            folder_embeddings = []

            # Process each image in the label folder
            for image_name in os.listdir(label_folder):
                image_path = os.path.join(label_folder, image_name)

                # Preprocess image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detect face
                bounding_boxes = detect_face(image)

                if len(bounding_boxes) > 0:
                    x1, y1, x2, y2 = bounding_boxes[0]
                    img_crop = image[y1:y2, x1:x2]
                    img_crop = normalize(img_crop)
                    img_crop = cv2.resize(img_crop, (160, 160))

                    vector_embedding = extract_feature(img_crop)
                    vector_embedding = vector_embedding.flatten()  # Flatten

                    folder_embeddings.append(vector_embedding)
                else:
                    print(f"No face detected in {image_name}")

            # Compute mean embedding for the label if embeddings exist
            if folder_embeddings:
                mean_embedding = np.mean(folder_embeddings, axis=0)
                embeddings.append(mean_embedding)
                labels.append(label)

    return np.array(embeddings), labels


def embeddings_to_csv(embeddings, labels, output_path='embedding_with_label.csv'):
    """
    Save embeddings and labels to CSV.

    Args:
        embeddings (numpy.ndarray): Array of embeddings.
        labels (list): List of labels.
        output_path (str): Output CSV path.
    """
    # Convert embeddings to string format
    formatted_embeddings = ['[' + ' '.join(map(str, emb)) + ']' for emb in embeddings]

    df = pd.DataFrame({'embedding': formatted_embeddings, 'person_name': labels})

    # Export to CSV without extra quotes
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONE, escapechar=' ')
















































