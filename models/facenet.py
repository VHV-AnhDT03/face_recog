"""
Model functions for face recognition using cosine similarity on embeddings.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine


def convert_str2array(input_str):
    """
    Convert string representation of array to numpy array.

    Args:
        input_str: String or array-like input.

    Returns:
        numpy.ndarray: Converted array.

    Raises:
        ValueError: If input type is unsupported.
    """
    if isinstance(input_str, str):
        input_str = input_str.strip('[]')
        number_list = [float(num) for num in input_str.split()]
    elif isinstance(input_str, (np.ndarray, list)):
        number_list = np.array(input_str, dtype=np.float32)
    else:
        raise ValueError("Unsupported input type: {}".format(type(input_str)))

    return np.array(number_list, dtype=np.float32)


def face_net(vector_c, embedding_df):
    """
    Recognize face by finding closest embedding.

    Args:
        vector_c (numpy.ndarray): Input embedding vector.
        embedding_df (pd.DataFrame): DataFrame with embeddings and labels.

    Returns:
        str: Recognized person name or unknown message.
    """
    distances = []
    
    # Convert string embeddings to arrays
    embedding_df['embedding'] = embedding_df['embedding'].apply(convert_str2array)

    # Calculate cosine distances to all stored embeddings
    for i in range(embedding_df.shape[0]):
        embedding_vector = embedding_df.iloc[i]['embedding']
        vector_c = np.ravel(vector_c)

        distances.append(cosine(embedding_vector, vector_c))

    # Find the minimum distance index
    min_index = np.argmin(distances)
    person_name = ""

    # Check if distance is below threshold
    if distances[min_index] > 0.3:
        person_name = "Khong xac dinh"
        print(f"NguongMin {distances[min_index]}")
    else:
        person_name = embedding_df.iloc[min_index, 1]
        print(f"Nguong Min {distances[min_index]} cua {person_name}")

    return person_name


