# Student Attention & Behavior Analysis System

This repository contains the complete codebase and MLOps deployment assets for the **Student Attention & Behavior Analysis System**. It leverages three deep learning models operating sequentially (student detection and tracking, behavior classification, and emotion/affective state estimation) to analyze classroom engagement and attention over time on a CPU-only environment.

---

## 1. System Architecture

```
Live Camera / Video ──> YOLOv8 Detector ──> ByteTrack (IDs) ──> Student Crops ──┬──> Behavior Model (MobileNetV3)
                                                                                └──> Emotion Model (MobileNetV3)
                                                                                           │
                                                                                           v
                                                                                   Attention Scoring &
                                                                                   Class Dashboard
```

The system coordinates three CPU-optimized models:
- **Detection & Tracking (YOLOv8 + ByteTrack)**: Detects every person in the frame, tracks them over time with stable IDs, and filters out the teacher using position/size heuristics.
- **Behavior Classifier (MobileNetV3 ONNX)**: Identifies postures and actions (e.g., writing, reading, raising hand, sleeping).
- **Emotion Classifier (MobileNetV3 ONNX)**: Estimates affective states from facial crops (e.g., engaged, confused, neutral, bored, distracted).

---

## 2. Directory Structure

The repository is organized as follows:
```
student-attention-system/
├── .gitignore               # Ignored weights, caches, and environments
├── Dockerfile               # Production Docker image configuration
├── README.md                # This manual
├── requirements.txt         # Dependencies for local Gradio application
├── src/                     # Core processing engine
│   ├── app.py               # Gradio dashboard application (Webcam & upload video)
│   ├── bytetrack_custom.yaml # Tracker configurations
│   └── pipeline.py          # Sequential modeling pipeline
├── models/                  # CPU-optimized model weights (Git-ignored)
│   ├── yolov8s.pt
│   ├── yolov8s.onnx
│   ├── behavior_best.onnx
│   ├── behavior_best.onnx.data
│   ├── behavior_class_names.json
│   ├── emotion_best.onnx
│   ├── emotion_best.onnx.data
│   └── emotion_class_names.json
├── deployment/              # FastAPI endpoints for MLOps compliance
│   ├── main.py              # FastAPI endpoints (/, /health, /predict)
│   └── requirements.txt     # Headless-optimized dependencies for Docker
├── kubernetes/              # Kubernetes manifests
│   ├── deployment.yaml      # Pod config with CPU/Memory bounds
│   └── service.yaml         # NodePort exposure
└── .github/
    └── workflows/
        └── ci.yml           # GitHub Actions syntax/lint and container build checks
```

---

## 3. Local Installation & Running (Gradio App)

### Prerequisites
- Python 3.10 or 3.11 installed.
- Pre-trained models placed in the `models/` directory (see model list above).

### Running the Gradio App
1. **Initialize Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\Activate.ps1
   ```
2. **Install Packages**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute**:
   ```bash
   python src/app.py
   ```
4. Navigate to `http://127.0.0.1:7860` in your web browser.

---

## 4. Serving FastAPI Locally

To test the API endpoint before deploying to AWS:
1. Activate your virtual environment and install the server requirements:
   ```bash
   pip install -r deployment/requirements.txt
   ```
2. Start the server using Uvicorn:
   ```bash
   uvicorn deployment.main:app --host 0.0.0.0 --port 8000
   ```
3. Open `http://localhost:8000/` and `http://localhost:8000/health` in a browser. Test posting a video to `http://localhost:8000/predict`.

---

## 5. Deployment on AWS EC2 (Ubuntu)

Follow the step-by-step commands to deploy the container and orchestrate it using Kubernetes:

### Step 5.1: Transfer Weights to AWS Instance
Since weights are git-ignored, upload them from your laptop directly to the EC2 server:
```bash
scp -i "attention-key.pem" -r models ubuntu@<YOUR-AWS-PUBLIC-IP>:~/student-attention-system/
```

### Step 5.2: Install Docker and Kubernetes Tools on AWS
Connect to AWS (`ssh -i "attention-key.pem" ubuntu@<PUBLIC-IP>`) and run:
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git curl unzip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

### Step 5.3: Containerize and Run on Docker
```bash
# Clone the repository
git clone <your-repository-url>
cd student-attention-system

# Build the docker container
docker build -t attention-system:v1 .

# Run the container
docker run -d -p 8000:8000 --name attention attention-system:v1
```

### Step 5.4: Run on Kubernetes (Minikube)
```bash
# Start Minikube cluster
minikube start --driver=docker

# Load the locally built image into Minikube's Docker runtime
minikube image load attention-system:v1

# Apply manifests
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml

# Verify state
kubectl get deployments
kubectl get pods
kubectl get services
```

### Step 5.5: Connect and Test
Query the Kubernetes NodePort service URL:
```bash
curl $(minikube service attention-service --url)/health
```
If connecting from outside, use SSH tunneling:
```bash
ssh -i attention-key.pem -L 8000:$(minikube ip):30080 ubuntu@<YOUR-AWS-PUBLIC-IP>
```
And navigate to `http://localhost:8000/health` on your host machine.
