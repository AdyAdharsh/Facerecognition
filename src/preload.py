# preload.py (Stored in ~/.deepface/weights/ directory)

import deepface
import os

print("--- Starting DeepFace Model Preloading ---")

# 1. Force download and initialization of FaceNet (for embeddings)
print("Loading FaceNet model (128-dimensional embeddings)...")
try:
    # Use dot notation for DeepFace submodules to avoid Pylance errors
    model_facenet = deepface.basemodels.FaceNet.loadModel() 
    print("✅ FaceNet loaded successfully.")
except Exception as e:
    print(f"❌ Error loading FaceNet: {e}")

# 2. Force download and initialization of MTCNN (CNN-based detection)
print("Loading MTCNN detector (CNN-based detection)...")
try:
    # Use dot notation for DeepFace submodules
    detector_mtcnn = deepface.detectors.FaceDetector.build_model('mtcnn') 
    print("✅ MTCNN loaded successfully.")
except Exception as e:
    print(f"❌ Error loading MTCNN: {e}")

print("--- Preloading Complete ---")