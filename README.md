# 🎯 Facerecognition: Real-Time Face Recognition Web Application

Live demo: https://facerecognition-533283114662.us-central1.run.app

Real-time face recognition with live enrollment, built for production-style deployment using Flask, DeepFace, and OpenCV. This project demonstrates an end-to-end pipeline: browser webcam → backend face detection & embedding → cosine-similarity recognition → live overlay and user registration — all containerized for Cloud Run.

⚠️ Important Notes
- First load may take 10–30 seconds due to ML model initialization (cold start) on serverless platforms.
- Face registrations are ephemeral on Cloud Run (stateless containers).
- Demo is intended for technical showcase; it does not provide persistent storage.

---

🚀 Key Highlights
Real-time, browser-driven face recognition with on-the-fly enrollment
- 🎥 Browser-based webcam streaming and overlay rendering
- 🧠 RetinaFace for detection, FaceNet (via DeepFace) for embeddings
- 🔐 Cosine similarity matching for recognition
- ➕ Live face enrollment from webcam (immediately available for recognition)
- 🐳 Fully containerized (Docker) and deployable to Google Cloud Run
- ⚙️ CPU-only inference to maximize portability across cloud runtimes

---

🏗️ What We Built
Core components and where to find them:

- app.py
  - Flask application exposing recognition & registration APIs and serving the UI
- src/detect.py
  - RetinaFace-based face detection and crop utilities
- src/embed.py
  - Embedding generation using FaceNet (via DeepFace)
- src/recognize.py
  - Matching logic, cosine similarity, thresholding, and result formatting
- src/register.py
  - Face enrollment logic: capture → embed → in-memory store
- src/utils.py
  - Helpers for embedding persistence (runtime), serialization, and utilities
- templates/index.html
  - Live recognition UI (webcam + overlay)
- templates/register.html
  - Face registration UI

Application flow (high level):
Browser Webcam → JavaScript (getUserMedia) → POST frame → /api/recognize_frame → RetinaFace detect → FaceNet embedding → Cosine similarity match → Result returned → Browser overlay

---

📦 Project Structure

The project structure is organized for clarity and ease of development. Below is a properly formatted tree view of the repository:

```text
Facerecognition/
├── app.py                  # Main Flask application (routes, APIs, server entrypoint)
├── Dockerfile              # Production Docker image definition
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and usage (this file)
├── src/                    # Application source code
│   ├── detect.py           # Face detection logic (RetinaFace wrapper, crop/align)
│   ├── embed.py            # Embedding generation (FaceNet / DeepFace wrapper)
│   ├── recognize.py        # Matching logic (cosine similarity, thresholds)
│   ├── register.py         # Face enrollment logic (capture → embed → in-memory store)
│   └── utils.py            # Utilities (serialization, persistence helpers, helpers)
└── templates/              # Frontend HTML templates
    ├── index.html          # Live recognition UI (webcam + overlay)
    └── register.html       # Face registration UI (enroll new users)
```

Notes:
- Keep application logic in src/ to separate implementation from entrypoints and deployment files.
- templates/ contains minimal frontend pages served by Flask; static assets (if any) can be added under a static/ directory.
- If you plan to add tests, create a tests/ directory at the repo root.

---

🚀 Quick Start

Clone and run locally
# Clone repository
git clone https://github.com/AdyAdharsh/Facerecognition.git
cd Facerecognition

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
# OR (production-like)
gunicorn --bind 0.0.0.0:9000 app:app

Open: http://localhost:9000

Docker
docker build -t facerecognition .
docker run -p 8080:8080 facerecognition

Cloud Run (example)
gcloud builds submit --tag gcr.io/<PROJECT_ID>/facerecognition
gcloud run deploy facerecognition \
  --image gcr.io/<PROJECT_ID>/facerecognition \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated

---

🧪 Runtime & Performance Notes
- CPU-only inference targeted for portability (no GPU required).
- Frame throttling is used on the client to limit backend load.
- Largest face selection heuristic for stable recognition in multi-face frames.
- Cosine similarity is used for embedding comparison; thresholds are configurable.

---

📊 Visualization & Demo
- Browser UI shows bounding boxes and labels directly over the live video feed.
- Registration UI lets users enter a name and enroll a face; the embedding is stored in-memory for the running container and is available immediately.
- No video frames are stored server-side; only embeddings exist in runtime memory.

---

🔬 Technical Details
Models & Libraries
- Face detection: RetinaFace (via DeepFace utility or custom wrapper)
- Embeddings: FaceNet (DeepFace)
- CV / utilities: OpenCV, NumPy
- Backend: Python 3.11, Flask, optionally Gunicorn for production
- Containerization: Docker, deployable to Google Cloud Run

Recognition pipeline
- Detect faces in an incoming frame
- Crop & align the largest face (configurable)
- Compute embedding with FaceNet
- Compare with registered embeddings using cosine similarity
- Return best match + confidence to the browser

Privacy & Security
- Webcam access handled purely in the browser (getUserMedia)
- No raw video is stored on the server
- Embeddings live only in-memory while the container instance runs
- Cloud Run deployment is stateless by design; persistent storage is not included in the demo

---

📁 Useful Commands & Scripts
- Start dev server: python app.py
- Start with Gunicorn: gunicorn --bind 0.0.0.0:9000 app:app
- Build Docker image: docker build -t facerecognition .
- Run Docker: docker run -p 8080:8080 facerecognition
- Deploy to Cloud Run: see Cloud Run example above

---

📝 Citation
If you use this project in research, please cite:
@software{facerecognition,
  title = {Facerecognition: Real-Time Face Recognition Web Application},
  author = {AdyAdharsh},
  year = {2024},
  url = {https://github.com/AdyAdharsh/Facerecognition},
  license = {MIT}
}

📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

---

🙏 Acknowledgments
- DeepFace and the authors of FaceNet & RetinaFace for the core models
- OpenCV and NumPy for image processing
- Google Cloud Run for easy deployment

🐛 Known Issues & Future Work
- Face registrations are ephemeral in the Cloud Run demo — add persistent storage (e.g., Cloud Storage / database)
- Improve multi-face handling and per-face tracking across frames
- Provide optional GPU support for lower latency on large-scale deployments
- Add authentication & secure enrollment workflow for production use
- Add automated tests and CI for model loading/health checks

💬 Contributing
Contributions are welcome — feel free to open an issue or submit a pull request.

📧 Contact
For questions or suggestions, please open an issue on GitHub: https://github.com/AdyAdharsh/Facerecognition/issues

Built with ❤️ using modern ML tooling and a focus on quick, portable demos.
