# Complete AWS & MLOps Deployment Architecture Guide

This document provides a comprehensive, step-by-step breakdown of the entire architecture, modifications, server resolutions, and execution procedures implemented for the **Student Attention System**.

---

## 🗺️ System Architecture Overview

The system operates as a hybrid cloud MLOps architecture:

```mermaid
graph TD
    subgraph Client Laptop (Local)
        A[Streamlit Dashboard UI: app_client.py] -->|1. Uploads Video File via HTTP POST| B(AWS EC2 Instance: 13.60.62.49)
        A2[Local Video File] --> A
    end

    subgraph AWS EC2 Host Instance (Ubuntu CPU)
        B -->|2. Maps incoming Port 8000| C[Background Port-Forward Daemon]
        C -->|3. Forwards traffic to NodePort 30080| D[Minikube Cluster Node]
        
        subgraph Kubernetes Namespace
            D -->|4. Routes to Service| E[Kubernetes Service: attention-service]
            E -->|5. Balances traffic to Port 8000| F[Kubernetes Pod: attention-deployment]
            
            subgraph Pod Container (attention-system:v1)
                F -->|6. Receives Request| G[FastAPI Router: main.py]
                G -->|7. Calls serving engine| H[Pipeline Engine: pipeline.py]
                H -->|8. YOLOv8 ONNX| I[Detection Mode]
                H -->|9. ByteTrack| J[Tracking Mode]
                H -->|10. Behavior ONNX| K[Behavior Mode]
                H -->|11. Emotion ONNX| L[Emotion Mode]
                H -->|12. Excludes Teacher| M[Filtering Logic]
            end
        end
    end
    
    G -->|13. Returns JSON Analytics Payload| A
```

---

## 📂 Directory Layout & Locations

| Component | Physical Path | Purpose / Description |
| :--- | :--- | :--- |
| **Gradio UI** | `src/app.py` | Local web app dashboard for webcam streaming and local video uploads. |
| **Pipeline Engine** | `src/pipeline.py` | Central Python engine coordinating YOLO detection, tracking, crop compilation, and ONNX classification. |
| **FastAPI Wrapper** | `deployment/main.py` | Exposes REST endpoints (`/`, `/health`, `/predict`) for cloud communication. |
| **Serving Requirements** | `deployment/requirements.txt` | Defines libraries (like `opencv-python-headless`) optimized for CPU serving. |
| **Kubernetes Pod Config** | `kubernetes/deployment.yaml` | Declares replication metrics, container ports, and pod resource bounds (1–3 Gi RAM). |
| **Kubernetes Service Config**| `kubernetes/service.yaml` | Declares NodePort mapping external port `30080` to internal container port `8000`. |
| **CI/CD Workflow** | `.github/workflows/ci.yml` | Automates docker packaging checks on GitHub runners. |
| **ONNX Models** | `models/` | Excluded from git. Holds ONNX weights and class configurations on both the laptop and AWS. |
| **Streamlit Client** | `app_client.py` | Client dashboard running on your laptop that talks to the remote AWS API. |

---

## 🛠️ Step-by-Step Implementation Details

### Phase 1: AWS EC2 Instance Creation & Setup
* **What was done**: Created an Ubuntu 26.04 LTS instance on AWS EC2 (`13.60.62.49`).
* **How it was done**:
  - Bound Security Group rules to open port 22 (SSH access) and port 8000 (FastAPI endpoints).
  - Generated `attention-key.pem` private key file to establish secure remote commands.
  - Linked host network aliases.

### Phase 2: Local Codebase Consolidation
* **What was done**: Restructured loose files from `Notebook 1-4 strcture` into a standardized directory structure.
* **How it was done**:
  - Extracted tracking parameters into `bytetrack_custom.yaml` inside `src/`.
  - Gathered classification weights and mapped label classes to structured JSON configurations.
  - Placed metrics profiling logs, epoch training charts, and confusion matrices inside `training/` subdirectories.

### Phase 3: Solving Server Environment Constraints
* **Issue 1: Host Python Version Incompatibility**
  - *Context*: The AWS Ubuntu host natively runs Python 3.14. Most deep learning wheels (PyTorch, ONNX, Ultralytics) do not support Python 3.14.
  - *Resolution*: Installed Miniconda and compiled an isolated Python 3.11.9 virtual environment inside `venv/`.
* **Issue 2: "Disk quota exceeded" (Errno 122) during pip install**
  - *Context*: Ubuntu limits `/tmp` to a memory-backed RAM disk of 3.8GB. Downloading and extracting PyTorch CUDA wheels (>4.5GB) crashed this space.
  - *Resolution*: Created a physical folder `~/tmp` on the server's root disk (75GB free space) and executed pip using `TMPDIR=~/tmp` to enforce physical disk caching.
* **Issue 4: Headless OpenCV Dependency Crash**
  - *Context*: Standard OpenCV requires host GUI display packages, throwing `libGL.so.1` missing errors.
  - *Resolution*: Installed `libgl1` and `libglib2.0-0` system libraries on the server using `apt`.

### Phase 4: Containerization with Docker
* **What was done**: Built a standalone container hosting the FastAPI server.
* **How it was done**:
  - Configured `Dockerfile` to use a Python 3.11 base image, copy source directories, and launch Uvicorn.
  - Built the container image on the host:
    ```bash
    sudo docker build -t attention-system:v1 .
    ```

### Phase 5: Kubernetes Pod Orchestration
* **What was done**: Set up Minikube and deployed containerized pods inside a Kubernetes cluster.
* **How it was done**:
  - Initialized Minikube using the Docker driver.
  - *Minikube OOM Fix*: Copying the 9.84GB image into Minikube exhausted the host's 8GB RAM and triggered the OOM killer. Resolved this by building the image directly inside Minikube's Docker runtime:
    ```bash
    eval $(minikube docker-env)
    docker build -t attention-system:v1 .
    ```
  - Deployed resources using manifests:
    ```bash
    kubectl apply -f kubernetes/
    ```
  - Forwarded cluster service NodePort (`30080`) to host public port `8000` via background daemon:
    ```bash
    nohup kubectl port-forward --address 0.0.0.0 service/attention-service 8000:8000 > port-forward.log 2>&1 &
    ```

### Phase 6: Streamlit Client UI Integration
* **What was done**: Wrote `app_client.py` to allow users to interact with the system visually from their laptops.
* **How it was done**:
  - Built a Streamlit interface that takes a video file upload.
  - Sends a multipart form request containing the binary video stream to the EC2 API `/predict` endpoint.
  - Parses the returned metrics JSON payload, rendering interactive charts (focus timeline, behavior frequency) and focus tables.
