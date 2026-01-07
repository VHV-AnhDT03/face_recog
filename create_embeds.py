"""
Script to create and save face embeddings from dataset images.
"""

from utils.embed import create_embedding, embeddings_to_csv

# Create embeddings from images in the Data folder
embeddings, labels = create_embedding('./Data')

# Save embeddings and labels to CSV file
embeddings_to_csv(embeddings, labels)