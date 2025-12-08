import cv2
from deepface import DeepFace

def detect_face(image, detector_type='cnn'):
    """
    Detects faces in the image using the specified method (MTCNN or Haar Cascade).
    
    Args:
        image (np.array): The input BGR image frame.
        detector_type (str): 'cnn' for MTCNN (default) or 'classical' for Haar Cascade.
        
    Returns:
        list: A list of dicts/tuples containing detected face info (bounding box, landmarks, confidence).
              Format: [{'box': (x, y, w, h), 'landmarks': {...}, 'confidence': float}]
    """
    
    # Map detector type to DeepFace backend name
    if detector_type == 'cnn':
        backend = 'mtcnn' # Multi-task Cascaded Convolutional Neural Network [cite: 21]
    elif detector_type == 'classical':
        backend = 'opencv' # DeepFace uses 'opencv' for Haar Cascade [cite: 26]
    else:
        return []

    results = []
    
    try:
        # DeepFace handles detection, alignment, and returns landmarks (for MTCNN)
        detected_faces = DeepFace.extract_faces(
            img_path=image, 
            detector_backend=backend, 
            enforce_detection=False # Allow processing even if no face is initially found
        )
        
        for face_info in detected_faces:
            x, y, w, h = face_info['facial_area'].values()
            
            # Note: DeepFace automatically handles alignment and cropping internally for embedding, 
            # but we return the raw box and a placeholder for structured output
            results.append({
                'box': (x, y, w, h),
                # Landmarks are useful for visualizing the alignment process
                'landmarks': face_info.get('landmarks', {}), 
                'confidence': face_info.get('confidence', 1.0)
            })

    except Exception as e:
        # print(f"Detection error with {detector_type}: {e}")
        pass

    return results