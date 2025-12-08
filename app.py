import streamlit as st
import cv2  # OpenCV for image processing
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, VideoProcessorBase
# NOTE: Make sure these core libraries are in your requirements.txt
# import deepface 
# import sklearn # if needed for recognition/clustering

# --- PLACEHOLDER IMPORTS (UNCOMMENT/ADJUST AS NEEDED) ---
# from src.detect import detect_faces
# from src.embed import get_embeddings
# from src.recognize import recognize_face
# from src.utils import LogManager 


# --- CONFIGURATION ---
# NOTE: Adjust these values based on your model/system performance
RECOGNITION_THRESHOLD = 0.6
FRAME_SKIP = 3  # Process every 3rd frame for performance


# --- VIDEO PROCESSING CLASS ---
# VideoTransformerBase handles receiving frames and sending them back
class FaceRecognitionTransformer(VideoTransformerBase):
    """
    A class that processes video frames in real-time for face recognition.
    """
    def __init__(self):
        # Initialize any models/trackers here to load them once
        # Example: self.detector = deepface.DeepFace.build_model("mtcnn")
        # Example: self.recognizer = load_your_recognizer_model()
        self.frame_count = 0
        self.detection_model = None # Placeholder
        self.recognition_model = None # Placeholder
        
        # Log manager placeholder
        # self.log_manager = LogManager() 
        
    def transform(self, frame: np.ndarray) -> np.ndarray:
        # Increment frame count
        self.frame_count += 1
        
        # Skip frames to reduce CPU load
        if self.frame_count % FRAME_SKIP != 0:
            return frame
        
        # Convert frame from BGR (OpenCV default) to RGB 
        img = frame.copy()
        
        # 1. Detect Faces (Placeholder Logic)
        # Replace with your actual detection function call
        # Example: faces = detect_faces(img, self.detection_model)
        
        # Placeholder: Assume one face in the middle for demonstration
        # In a real app, you'd get (x, y, w, h) for all faces
        h, w, _ = img.shape
        faces = [(w//4, h//4, w//2, h//2)] 

        for (x, y, w, h) in faces:
            # 2. Recognize Face (Placeholder Logic)
            # Example: recognized_name, score = recognize_face(img, x, y, w, h, self.recognition_model)
            recognized_name = "Unknown" # Placeholder result
            score = 0.0
            
            # --- Decision and Visualization ---
            
            if recognized_name != "Unknown" and score >= RECOGNITION_THRESHOLD:
                color = (0, 255, 0) # Green for known user
                # self.log_manager.log_access(recognized_name)
            else:
                color = (0, 0, 255) # Red for unknown user
                recognized_name = "Unknown" 
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{recognized_name}: {score:.2f}"
            cv2.putText(img, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return img

# --- STREAMLIT UI ---

def main():
    global RECOGNITION_THRESHOLD
    
    st.title("Smart Office Face Recognition System 📸")
    st.sidebar.title("Configuration")

    # Sidebar control for threshold
    RECOGNITION_THRESHOLD = st.sidebar.slider(
        "Recognition Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05
    )

    # Start the WebRTC Streamer
    webrtc_streamer(
        key="face-recognition-stream",
        # Use SENDRECV mode for two-way communication (video in, video out)
        mode=WebRtcMode.SENDRECV,
        
        # --- CRITICAL FIX: Enhanced STUN/TURN configuration to fix aioice errors ---
        rtc_configuration={
            "iceServers": [
                # Google's public STUN server (standard)
                {"urls": ["stun:stun.l.google.com:19302"]},
                # Mozilla's public STUN server (as backup)
                {"urls": ["stun:stun.services.mozilla.com"]}
            ]
        },
        video_transformer_factory=FaceRecognitionTransformer,
        async_transform=True
    )

    st.markdown("---")
    st.subheader("Access Log (Placeholder)")
    # Placeholder for displaying logs
    # if st.button("Refresh Log"):
    #     st.dataframe(LogManager().get_logs())

# --- EXECUTION ---
if __name__ == "__main__":
    main()