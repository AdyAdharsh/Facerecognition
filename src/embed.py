import numpy as np
from deepface import DeepFace

# Global variables to hold the model configuration
MODEL_NAME = "Facenet"
DIMENSIONS = 128

def get_embedding(face_image):
    """
    Extracts the 128-dimensional embedding from the face image using FaceNet.
    
    Args:
        face_image (np.array): The input BGR image frame (must contain a face).
        
    Returns:
        np.array or None: The 128-dimensional embedding vector.
    """
    try:
        # DeepFace handles alignment, preprocessing, and model prediction internally.
        # Ensure only the area containing the face is passed, or let DeepFace handle cropping.
        
        # We use a wrapper function to ensure only the embedding is returned
        embedding_objs = DeepFace.represent(
            img_path=face_image,
            model_name=MODEL_NAME, 
            enforce_detection=False # If face is already pre-cropped
        )
        
        if embedding_objs:
            # The embedding is a 128-D vector 
            embedding = embedding_objs[0]["embedding"]
            return np.array(embedding)
            
    except Exception as e:
        # print(f"Embedding generation error: {e}")
        return None
        
    return None