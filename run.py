# run.py

import os
# --- Environment Configuration (CRITICAL for Stability) ---
# Forces TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
# Limit threads for NumPy/NumExpr
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

# CRITICAL FIX for macOS mutex/threading conflicts, often related to OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
# ----------------------------------------------------------

import argparse
import cv2
import sys
import numpy as np
import tensorflow as tf # Required for DeepFace's underlying model

# --- Set OpenCV Threads (Recommended for Stability on macOS) ---
# Setting to 0 tells OpenCV to use the default platform-specific thread management, 
# which often resolves conflicts with other threaded libraries like TensorFlow.
cv2.setNumThreads(0) 
# ------------------------------------------------------

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
        
        # Ensure bounding box coordinates are non-negative and within frame bounds
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        
        # Extract the face region
        face_img = frame[y:y+h, x:x+w] 
        
        # Draw bounding box on the displayed frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        if mode == 'register':
            # Registration Logic
            name = input("Enter name to register: ").strip()
            
            if not name:
                return "❌ Registration cancelled. Name cannot be empty."
            
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
            
            # Determine color for the bounding box and text
            if name != "Unknown":
                color = (0, 255, 0) # Green for recognized
                label = f"{name} (D:{distance:.2f})"
                result_msg = f"✅ Recognized: {name} (Dist: {distance:.2f})"
            else:
                color = (0, 0, 255) # Red for unknown
                label = f"Unknown (D:{distance:.2f})"
                result_msg = f"⚠️ Unknown Person (Dist: {distance:.2f})"

            # Draw recognition result on the frame
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            return result_msg
    
    return "No face detected in the frame."

# --- Main CLI Function ---
def main():
    parser = argparse.ArgumentParser(description="Smart Office Face Recognition System CLI.")
    parser.add_argument('--mode', required=True, choices=['register', 'recognize'], help="Operation mode: 'register' a new face, or 'recognize' an existing one.")
    parser.add_argument('--detector', required=False, default='cnn', choices=['cnn', 'classical'], help="Detector backend: cnn (MTCNN) or classical (Haar Cascade). Defaults to 'cnn'.")
    
    args = parser.parse_args()
    data_store = load_embeddings() # Initial load of persistent data

    cap = cv2.VideoCapture(0) # 0 is typically the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        # Attempt to use a different camera index if 0 fails
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
             print("Fatal Error: Could not open any webcam device.")
             sys.exit(1)
        
    print(f"\n--- Running in {args.mode.upper()} mode with {args.detector.upper()} detector. ---")
    print(f"Loaded {len(data_store['names'])} registered users.")
    print("Press 'c' to CAPTURE/PROCESS a face or 'q' to QUIT.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Display instructions on the frame
        cv2.putText(frame, f"MODE: {args.mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'c' to capture | 'q' to quit.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Smart Office Face Recognition CLI', frame)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            print("-" * 30)
            print(f"Capture command received. Processing frame in {args.mode} mode...")
            
            # Process the captured frame
            # NOTE: We pass a copy of the frame to prevent conflicts with the display loop
            result = process_frame(frame.copy(), args.mode, args.detector, data_store)
            print(result)

            # If a new user was registered, immediately reload the data store 
            if "Registration successful" in result:
                data_store = load_embeddings() 
                print(f"Data store reloaded. Total users: {len(data_store['names'])}")
            print("-" * 30)


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()