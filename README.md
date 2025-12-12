🎯 Real-Time Face Recognition Web Application

A production-style real-time face recognition system built with Flask, DeepFace, and OpenCV, featuring live webcam face detection, recognition, and on-the-fly user registration, fully containerized and deployed on Google Cloud Run.

This project demonstrates computer vision, machine learning inference, backend APIs, frontend integration, and cloud deployment in a single end-to-end system.

🚀 Live Demo

🔗 Cloud Run URL
https://facerecognition-533283114662.us-central1.run.app

⚠️ Important Notes
	•	First load may take 10–30 seconds due to ML model initialization (cold start).
	•	Face registrations are ephemeral (Cloud Run containers are stateless).
	•	Demo is intended for technical showcase, not persistent storage.

  ✨ Key Features
	•	🎥 Browser-based webcam streaming
	•	🧠 Face detection using RetinaFace
	•	🔐 Face embeddings generated with FaceNet
	•	🆔 Real-time face recognition via cosine similarity
	•	➕ Live face enrollment directly from webcam
	•	🎨 Bounding boxes and labels rendered on video feed
	•	☁️ Cloud-native deployment using Docker & Cloud Run

  🛠 Tech Stack

Backend
	•	Python 3.11
	•	Flask
	•	Gunicorn

Computer Vision / ML
	•	DeepFace
	•	FaceNet
	•	RetinaFace
	•	OpenCV
	•	NumPy

Frontend
	•	HTML
	•	CSS
	•	JavaScript (Web APIs)

Cloud & DevOps
	•	Docker
	•	Google Cloud Run
	•	Google Container Registry

  Browser Webcam
     ↓
JavaScript (getUserMedia)
     ↓
Flask API (/api/recognize_frame)
     ↓
Face Detection (RetinaFace)
     ↓
Embedding Generation (FaceNet)
     ↓
Cosine Similarity Matching
     ↓
Result → Browser Overlay

📸 Application Flow

1️⃣ Face Recognition
	1.	Browser captures webcam frames
	2.	Frames sent to backend every few seconds
	3.	Face detected and cropped
	4.	Embedding generated
	5.	Compared against registered users
	6.	Name + confidence displayed on video

2️⃣ Face Registration
	1.	User enters name
	2.	Captures face via webcam
	3.	Embedding generated and stored
	4.	Available immediately for recognition

📂 Project Structure
  Facerecognition/
├── app.py                  # Main Flask application
├── Dockerfile               # Production Docker image
├── requirements.txt         # Python dependencies
├── README.md
├── src/
│   ├── detect.py            # Face detection logic
│   ├── embed.py             # Face embedding generation
│   ├── recognize.py         # Matching logic
│   ├── register.py          # Face enrollment logic
│   └── utils.py             # Embedding persistence helpers
└── templates/
    ├── index.html            # Live recognition UI
    └── register.html         # Face registration UI

🧪 Local Development

1️⃣ Create virtual environment

python -m venv .venv
source .venv/bin/activate

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Run locally

python app.py
# OR
gunicorn --bind 0.0.0.0:9000 app:app

Open:

http://localhost:9000


🐳 Docker Build & Run

docker build -t facerecognition .
docker run -p 8080:8080 facerecognition

☁️ Cloud Deployment (Google Cloud Run)

gcloud builds submit --tag gcr.io/<PROJECT_ID>/facerecognition
gcloud run deploy facerecognition \
  --image gcr.io/<PROJECT_ID>/facerecognition \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated

  
⚙️ Performance & Design Decisions
	•	CPU-only inference for portability
	•	Frame throttling to reduce load
	•	Largest face selection for stability
	•	Cosine similarity for embedding comparison
	•	Stateless container design (Cloud Run best practice)

🔐 Privacy & Security
	•	Webcam access handled entirely in-browser
	•	No video stored on server
	•	Face embeddings exist only during runtime
	•	No personal data persistence in cloud deployment  
    
