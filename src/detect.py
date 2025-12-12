import numpy as np
from deepface import DeepFace

def detect_face(img_rgb, detector_type="opencv"):
    """
    Detect faces using DeepFace extract_faces() with OpenCV backend.
    This backend does NOT require downloading any heavy model files,
    so it works reliably inside Cloud Run.
    """
    try:
        detections = DeepFace.extract_faces(
            img_path=img_rgb,
            detector_backend="opencv",
            enforce_detection=False,
            align=False
        )
    except Exception as e:
        print("Detection error:", e)
        return []

    faces = []
    for det in detections:
        area = det.get("facial_area")
        if not area:
            continue

        faces.append({
            "box": [
                int(area["x"]),
                int(area["y"]),
                int(area["w"]),
                int(area["h"])
            ]
        })

    return faces