# run.py
# run.py (Add this to the very top, before imports)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Forces TensorFlow to use CPU only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Suppress warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" # For GPU, but harmless on CPU
os.environ["NUMEXPR_NUM_THREADS"] = "1" # Limit threads for NumPy/NumExpr
import argparse
import cv2
import sys
import numpy as np
import tensorflow as tf
# Import your core logic modules
from src.utils import load_embeddings
from src.detect import detect_face
from src.embed import get_embedding
from src.recognize import recognize_face_by_embedding
from src.register import register_new_user 

# Define the logic to process a captured frame for registration or recognition
def process_frame(frame, mode, detector_key, data_store):
    
    # 1. Detection
    detected_faces = detect_face(frame, detector_type=detector_key)
    
    if detected_faces:
        # Focus on the largest face for processing
        main_face = max(detected_faces, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = main_face['box']
        face_img = frame[y:y+h, x:x+w] 
        
        # Draw bounding box on the displayed frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        if mode == 'register':
            # Ask for name when face is captured
            name = input("Enter name to register: ")
            
            # 2. Registration Logic
            if register_new_user(face_img, name):
                return f"✅ Registration successful for {name}! Please run recognize mode next."
            else:
                return "❌ Registration failed. Could not generate embedding."
                
        elif mode in ['recognize', 'identify']:
            # 3. Recognition Logic
            embedding = get_embedding(face_img)
            
            if embedding is None:
                return "⚠️ Could not generate embedding for recognition."

            name, distance = recognize_face_by_embedding(
                embedding, 
                data_store['embeddings'], 
                data_store['names']
            )
            
            if name != "Unknown":
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                return f"✅ Recognized: {name} (Dist: {distance:.2f})"
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return f"⚠️ Unknown Person (Dist: {distance:.2f})"
    
    return "No face detected in the frame."

# --- Main CLI Function ---
def main():
    parser = argparse.ArgumentParser(description="Smart Office Face Recognition System CLI.")
    parser.add_argument('--mode', required=True, choices=['register', 'recognize', 'identify'], help="Operation mode (register or identify/recognize).")
    parser.add_argument('--detector', required=True, choices=['cnn', 'classical'], help="Detector backend: cnn (MTCNN) or classical (Haar Cascade).")
    
    args = parser.parse_args()
    data_store = load_embeddings() # Initial load of persistent data

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
        
    print(f"\n--- Running in {args.mode.upper()} mode with {args.detector.upper()} detector. ---")
    print("Press 'c' to CAPTURE/PROCESS a face or 'q' to QUIT.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display instructions on the frame
        cv2.putText(frame, f"MODE: {args.mode.upper()} | DETECTOR: {args.detector.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'c' to capture.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Smart Office Face Recognition CLI', frame)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            print(f"Capture command received. Processing frame in {args.mode} mode...")
            
            # Process the captured frame
            result = process_frame(frame, args.mode, args.detector, data_store)
            print(result)

            # FIX: Pylance Scope Fix - If registered, reload the local data_store variable for immediate recognition.
            if "Registration successful" in result:
                data_store = load_embeddings() 
                print("Data store reloaded for new user recognition.")


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()