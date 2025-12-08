import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# The user is identified if cosine distance < 0.5. 
# Cosine Similarity = 1 - Cosine Distance
SIMILARITY_THRESHOLD = 0.5 

def recognize_face_by_embedding(current_embedding, known_embeddings, known_names):
    """
    Identifies a known person by comparing the current face embedding against stored ones.
    
    Args:
        current_embedding (np.array): The 128-D embedding of the live face.
        known_embeddings (list): List of stored embedding vectors (numpy arrays).
        known_names (list): List of names corresponding to the embeddings.
        
    Returns:
        tuple: (Identified Name or "Unknown", Cosine Distance)
    """
    if not known_embeddings or current_embedding is None:
        return "Unknown", 1.0 # No users registered or failed embedding
        
    # Reshape for comparison: (1, 128)
    current_embedding = current_embedding.reshape(1, -1)
    
    # Convert list of known embeddings to a NumPy array (N, 128)
    known_embeddings_array = np.array(known_embeddings)

    # Calculate Cosine Similarity (vector comparison)
    similarities = cosine_similarity(current_embedding, known_embeddings_array)[0]
    
    # Find the best match (highest similarity)
    best_match_index = np.argmax(similarities)
    best_similarity = similarities[best_match_index]
    
    # Calculate distance for checking the threshold: Distance = 1 - Similarity
    best_distance = 1.0 - best_similarity
    
    # Check if the similarity surpasses the threshold (i.e., distance < 0.5)
    if best_distance < SIMILARITY_THRESHOLD:
        identified_name = known_names[best_match_index]
        return identified_name, best_distance
    else:
        # System notes the person as unknown [cite: 44]
        return "Unknown", best_distance