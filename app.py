import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np

# --- Import your modular code ---
from src.utils import load_embeddings, save_embeddings
from src.detect import detect_face
from src.embed import get_embedding
from src.recognize import recognize_face_by_embedding
from src.register import register_new_user # Ensure this is in your src folder

# Use Streamlit's cache to load the persistent data once
@st.cache_resource
def load_registered_data():
    """Load the persistent data structure."""
    return load_embeddings()

# Global variables for capturing a registration frame
REGISTRATION_FRAME = None
# Initialize frame_lock in session state if it doesn't exist
if 'frame_lock' not in st.session_state:
    st.session_state['frame_lock'] = False
FRAME_LOCK = st.session_state['frame_lock']

class FaceRecognitionTransformer(VideoTransformerBase):
    def __init__(self, data_store, detector_key, mode):
        self.data_store = data_store
        self.detector_key = detector_key
        self.mode = mode
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr")

        # 1. Detection
        detected_faces = detect_face(img, detector_type=self.detector_key)
        
        if detected_faces:
            # We focus on the largest face for simplicity
            main_face = max(detected_faces, key=lambda x: x['box'][2] * x['box'][3])
            x, y, w, h = main_face['box']
            
            # Crop the face for processing (Alignment & Cropping) 
            face_img = img[y:y+h, x:x+w] 
            
            # 2. Draw bounding box and label
            label = "Processing..."
            color = (255, 255, 0) # Yellow/Cyan
            
            if self.mode == "Recognition":
                # 3. Recognition Logic
                
                # Embedding extraction
                embedding = get_embedding(face_img)

                # Comparison Logic
                name, distance = recognize_face_by_embedding(
                    embedding, 
                    self.data_store['embeddings'], 
                    self.data_store['names']
                )

                label = f"{name} (Dist: {distance:.2f})"
                if name != "Unknown":
                    color = (0, 255, 0) # Green for known
                else:
                    color = (0, 0, 255) # Red for unknown
            
            elif self.mode == "Registration":
                # 3. Registration capture mode
                label = "Ready to Register"
                color = (0, 255, 255) # Yellow
                
                # Capture the frame for registration if the lock is not set
                global REGISTRATION_FRAME
                if not st.session_state['frame_lock']:
                    REGISTRATION_FRAME = face_img
            
            # Draw the box and text
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img

# --- Streamlit UI ---

st.set_page_config(page_title="Smart Office Face Recognition", layout="wide")
# Removed erroneous tag
st.title("👨‍💼 Smart Office Access System")
st.markdown("This system identifies staff members using a webcam, welcomes them, or registers new visitors.")

st.sidebar.title("System Controls")
mode = st.sidebar.radio("Select Mode", ["Recognition", "Registration"])
detector_type = st.sidebar.radio(
    "Select Detector (Bonus Feature)", 
    ["CNN-based (Default)", "Classical (Haar Cascade)"]
)
detector_key = 'cnn' if detector_type == 'CNN-based (Default)' else 'classical'

# Load the persistent data
data_store = load_registered_data()

if mode == "Registration":
    st.header("📝 New User Registration")
    st.info("Press 'Start' to activate the camera. When ready and a face is detected, enter the staff name and press 'Capture & Register'.")
    
    user_name = st.text_input("Enter Staff Name", key='reg_name')
    
    # ------------------ Registration Stream ------------------
    webrtc_ctx_reg = webrtc_streamer(
        key="registration_stream",
        video_transformer_factory=lambda: FaceRecognitionTransformer(data_store, detector_key, mode),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
    
    if st.button("Capture & Register") and webrtc_ctx_reg.state.playing:
        if not user_name:
            st.error("Please enter a name for registration.")
        elif REGISTRATION_FRAME is None:
            st.error("No face detected in the frame. Please look at the camera.")
        else:
            # Set lock to prevent transformer from overwriting REGISTRATION_FRAME
            st.session_state['frame_lock'] = True 
            
            with st.spinner(f"Registering {user_name} with {detector_type} detector..."):
                if register_new_user(REGISTRATION_FRAME, user_name):
                    st.success(f"Registration successful for **{user_name}**! Embedding stored to {{data/embeddings.pkl}}")
                    # Force reload the data store to include the new user
                    load_registered_data.clear() 
                    st.session_state['frame_lock'] = False
                else:
                    st.error("Registration failed. Could not generate embedding.")

elif mode == "Recognition":
    st.header("🔑 Real-time Recognition Check")
    st.write(f"Currently **{len(data_store['embeddings'])}** users are registered.")

    # ------------------ Recognition Stream ------------------
    webrtc_ctx_rec = webrtc_streamer(
        key="recognition_stream",
        video_transformer_factory=lambda: FaceRecognitionTransformer(data_store, detector_key, mode),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
    
    # ✅ FIX: IndentationError is fixed here by indenting the st.success line.
    if webrtc_ctx_rec.state.playing:
        st.success(f"Recognition running with **{detector_type}** detector...")