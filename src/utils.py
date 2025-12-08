import pickle
import os

# Define the persistent storage path, navigating two levels up from src/
EMBEDDINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'embeddings.pkl')

def load_embeddings():
    """
    Loads all registered embeddings and associated names from the pickle file.
    
    Returns:
        dict: A dictionary with 'embeddings' (list of numpy arrays) and 'names' (list of strings).
    """
    if not os.path.exists(EMBEDDINGS_PATH):
        # Initialize an empty structure if the file doesn't exist
        return {'embeddings': [], 'names': []}
    
    try:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            data = pickle.load(f)
            # Ensure the loaded data has the expected structure
            if 'embeddings' not in data or 'names' not in data:
                 return {'embeddings': [], 'names': []}
            return data
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        # In case of corruption, return empty data
        return {'embeddings': [], 'names': []}

def save_embeddings(data):
    """
    Saves the updated embeddings and names back to the pickle file.
    
    Args:
        data (dict): The dictionary containing 'embeddings' and 'names' lists.
    """
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    
    try:
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        return False