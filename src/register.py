from .embed import get_embedding
from .utils import load_embeddings, save_embeddings

def register_new_user(face_image, name):
    """
    Registers a new user by generating an embedding and persisting it.
    
    Args:
        face_image (np.array): The input BGR image frame (must contain a face).
        name (str): The name of the user to register.
        
    Returns:
        bool: True on successful registration, False otherwise.
    """
    embedding = get_embedding(face_image)
    
    if embedding is not None:
        data = load_embeddings()
        
        # Store the created embedding and name
        data['embeddings'].append(embedding)
        data['names'].append(name)
        
        # Persist data to {data/embeddings.pkl}
        return save_embeddings(data)
    
    return False