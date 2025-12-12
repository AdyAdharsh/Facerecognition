import os

# --- Environment Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # Force CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ---------------------------------

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

from src.utils import load_embeddings
from src.detect import detect_face
from src.embed import get_embedding
from src.recognize import recognize_face_by_embedding
from src.register import register_new_user

app = Flask(__name__)
app.secret_key = "your_super_secret_key_here"


# ------------------------
# Routes (Pages)
# ------------------------

@app.route("/")
def index():
    """Main recognition UI page."""
    data = load_embeddings()
    num_users = len(data["names"])
    return render_template("index.html", num_users=num_users)


@app.route("/register")
def register_page():
    """Registration page UI."""
    data = load_embeddings()
    num_users = len(data["names"])
    return render_template("register.html", num_users=num_users)


# ------------------------
# API: Frame Recognition
# ------------------------

@app.route("/api/recognize_frame", methods=["POST"])
def recognize_frame():
    """Processes webcam frame and returns recognition match."""
    if "frame" not in request.files:
        return jsonify({"success": False, "error": "No frame uploaded"}), 400

    file_bytes = np.frombuffer(request.files["frame"].read(), np.uint8)
    frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return jsonify({"success": False, "error": "Failed to decode image"}), 400

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    detected_faces = detect_face(frame_rgb, detector_type="retinaface")
    if not detected_faces:
        return jsonify({"success": True, "found": False, "message": "No face detected"})

    # Pick the largest detected face
    main_face = max(detected_faces, key=lambda x: x["box"][2] * x["box"][3])
    x, y, w, h = main_face["box"]

    h_img, w_img = frame_bgr.shape[:2]
    x = max(0, x); y = max(0, y)
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))

    face_img = frame_bgr[y:y+h, x:x+w]

    embedding = get_embedding(face_img)
    if embedding is None:
        return jsonify({
            "success": True,
            "found": False,
            "message": "Face detected but embedding failed",
            "box": [x, y, w, h]
        })

    data = load_embeddings()
    known_embeddings = data["embeddings"]
    known_names = data["names"]

    if not known_embeddings:
        return jsonify({
            "success": True,
            "found": False,
            "message": "No registered users yet",
            "box": [x, y, w, h]
        })

    name, distance = recognize_face_by_embedding(embedding, known_embeddings, known_names)

    return jsonify({
        "success": True,
        "found": True,
        "name": name,
        "distance": float(distance),
        "box": [x, y, w, h]
    })


# ------------------------
# API: Register New Face
# ------------------------

@app.route("/api/register_frame", methods=["POST"])
def register_frame():
    """Registers a new user from webcam frame."""
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400

    if "frame" not in request.files:
        return jsonify({"success": False, "error": "No frame uploaded"}), 400

    file_bytes = np.frombuffer(request.files["frame"].read(), np.uint8)
    frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return jsonify({"success": False, "error": "Failed to decode image"}), 400

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detected_faces = detect_face(frame_rgb, detector_type="retinaface")

    if not detected_faces:
        return jsonify({"success": False, "error": "No face detected"})

    main_face = max(detected_faces, key=lambda x: x["box"][2] * x["box"][3])
    x, y, w, h = main_face["box"]

    h_img, w_img = frame_bgr.shape[:2]
    x = max(0, x); y = max(0, y)
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))

    face_img = frame_bgr[y:y+h, x:x+w]

    success = register_new_user(face_img, name)

    if not success:
        return jsonify({"success": False, "error": "Embedding generation failed"})

    return jsonify({"success": True, "message": f"Registration successful for {name}!"})


# ------------------------
# Health Check
# ------------------------

@app.route("/healthz")
def healthz():
    return "ok", 200


if __name__ == "__main__":
    print("Running locally at http://127.0.0.1:9000/")
    app.run(host="0.0.0.0", port=9000, debug=True)