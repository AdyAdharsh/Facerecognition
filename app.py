import streamlit as st
import cv2  # OpenCV for image processing
import numpy as np
import av  # REQUIRED: Must be in requirements.txt (pip install av)
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode 

# --- PLACEHOLDER IMPORTS (Adjust as needed) ---
# NOTE: The logic here is temporarily simplified to test WebRTC stability.
# The heavy imports (deepface, etc.) should be outside the cached function.

# --- CONFIGURATION ---
RECOGNITION_THRESHOLD = 0.6
FRAME_SKIP = 3


# --- CRITICAL FIX 1: CACHE THE HEAVY MODELS SEPARATELY ---
@st.cache_resource
def load_deepface_models():
    """Loads all heavy models only once, safely outside the thread."""
    # NOTE: Keep your actual heavy model loading logic here (e.g., DeepFace.build_model)
    # This function is retained to verify if the issue is in the loading itself.
    return "LOADED_MODELS_PLACEHOLDER" 


# --- CRITICAL FIX 2: CACHE THE FACTORY ---
@st.cache_resource
def get_face_processor_factory():
    """Returns the processor class, ensuring its initialization is thread-safe."""
    
    class FaceRecognitionProcessor(VideoProcessorBase):
        """Processes video frames using the standard VideoProcessorBase."""
        def __init__(self):
            # We call the model loading function, but we won't use the result in recv
            # This tests if the memory consumption is the issue.
            self.models_placeholder = load_deepface_models() 
            self.frame_count = 0
            
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # --- SIMPLIFIED PROCESSING LOGIC (TEMPORARY DEBUGGING) ---
            
            # Convert frame from av.VideoFrame to numpy array (BGR)
            img = frame.to_ndarray(format="bgr24")
            
            # Simple, lightweight OpenCV draw operation for verification
            h, w, _ = img.shape
            cv2.putText(img, "WebRTC OK", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Convert back to av.VideoFrame before returning
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    return FaceRecognitionProcessor

# --- STREAMLIT UI ---

def main():
    global RECOGNITION_THRESHOLD
    
    st.title("Smart Office Face Recognition System 📸")
    st.sidebar.title("Configuration")

    RECOGNITION_THRESHOLD = st.sidebar.slider(
        "Recognition Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05
    )

    # --- FINAL WEBRTC STREAMER CALL ---
    webrtc_streamer(
        key="face-recognition-stream-final-cache",  
        mode=WebRtcMode.SENDRECV,
        
        # --- Enhanced STUN/TURN configuration to stabilize cloud connection ---
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun.services.mozilla.com"]}
            ]
        },
        
        # Use the CACHED factory function
        video_processor_factory=get_face_processor_factory,  
        async_processing=True                                
    )

    st.markdown("---")
    st.subheader("Access Log (Placeholder)")

# --- EXECUTION ---
if __name__ == "__main__":
    main()