import cv2
from deepface import DeepFace

def get_embedding(face_bgr):
    """
    Generate embedding using DeepFace (Facenet model).
    model_name='Facenet' is supported and cached automatically.
    """
    try:
        # DeepFace expects a 160x160 RGB image for Facenet
        img = cv2.resize(face_bgr, (160, 160))

        embedding = DeepFace.represent(
            img_path=img,
            model_name="Facenet",
            detector_backend="skip"  # skip detection because we already cropped
        )[0]["embedding"]

        return embedding

    except Exception as e:
        print("Embedding error:", e)
        return None