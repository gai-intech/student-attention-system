# Viva Presentation & Evidence Guide

This guide contains the exact commands, scripts, and checkpoints to present your project during the evaluation (viva). It walks you through showing evidence for every stage of your AWS and MLOps deployment.

---

## 💻 Presentation Cheat Sheet: Step-by-Step Commands

### Step 1: Prove Your AWS EC2 Server is Active
Show that you are connected to a remote cloud machine on AWS.
1. Open a terminal on your laptop and run:
   ```bash
   ssh -i "attention-key.pem" ubuntu@13.60.62.49
   ```
2. Once logged in, show the system details and architecture:
   ```bash
   uname -a
   lsb_release -a
   uptime
   ```
   * **What this proves**: Proves your host is a remote Ubuntu 26.04 server hosted on an AWS cloud instance, running continuously.

---

### Step 2: Show the Git Code Structure & Remote Repository
Show that your project follows standardized production code layouts.
1. Run `git status` or show the files on the server:
   ```bash
   cd ~/student-attention-system
   ls -la
   ```
2. Show the active remote repository:
   ```bash
   git remote -v
   ```
   * **What this proves**: Proves your codebase is synced with user version control systems at `https://github.com/gai-intech/student-attention-system.git`.

---

### Step 3: Verify the Isolated Python 3.11.9 Environment
Show that you resolved the Python version compatibility issues.
1. Check the active python version in the virtual environment:
   ```bash
   ~/student-attention-system/venv/bin/python --version
   ```
2. Verify that all required serving libraries (FastAPI, ONNX Runtime, Ultralytics) are installed inside the venv:
   ```bash
   ~/student-attention-system/venv/bin/pip list | grep -E "fastapi|onnxruntime|ultralytics|torch"
   ```
   * **What this proves**: Proves that you bypassed the host's incompatible Python 3.14 shell by compiling an isolated Python 3.11.9 virtual environment containing all required serving weights.

---

### Step 4: Verify the Headless System Libraries
Show that the server has correct graphics configurations for processing video headless.
1. Verify that `libgl1` and `libglib2.0-0` are installed:
   ```bash
   dpkg -l | grep -E "libgl1|libglib2"
   ```
2. Test import compatibility of OpenCV in python:
   ```bash
   ~/student-attention-system/venv/bin/python -c "import cv2; print('OpenCV Loaded successfully:', cv2.__version__)"
   ```
   * **What this proves**: Proves that you successfully resolved the headless container graphics problem (`libGL.so.1` crash) by configuring the system graphics runtime libraries.

---

### Step 5: Verify the Standalone Docker Engine
Show that your system is fully containerized.
1. Show that the Docker daemon is active and configured:
   ```bash
   docker --version
   sudo docker ps -a
   ```
2. Show the built local container images:
   ```bash
   sudo docker images
   ```
   * **What this proves**: Proves your pipeline builds successfully into an immutable Docker image (`attention-system:v1`), ready for deployment to any server.

---

### Step 6: Verify Kubernetes Cluster & Pod Orchestration
Show the enterprise-level orchestration running under Minikube.
1. Show the status of the Minikube cluster:
   ```bash
   minikube status
   ```
2. Show that your Kubernetes deployment and services are fully active:
   ```bash
   kubectl get deployments
   kubectl get services
   kubectl get pods -o wide
   ```
   * **What this proves**: Proves your system is managed by a local Kubernetes scheduler, demonstrating pod routing, health tracking, and system resource limits.

---

### Step 7: Verify Port Forwarding & Public Endpoint Access
Show how external clients access the Kubernetes services.
1. Show that port forwarding is active in the background:
   ```bash
   ps aux | grep port-forward
   ```
2. Print the logs of the background port-forwarding process:
   ```bash
   cat ~/port-forward.log
   ```
3. Run a local health check directly on the host:
   ```bash
   curl -s http://localhost:8000/health
   ```
   * **What this proves**: Proves that incoming traffic on public port 8000 is safely forwarded into the internal Kubernetes cluster service mapping.

---

## 🛠️ Live Demonstration of the API

Run this live during your viva to show the system processing input data:

1. **Start the local Streamlit Client on your laptop**:
   ```bash
   streamlit run app_client.py
   ```
2. **Open the browser** to `http://localhost:8501`.
3. Show the **Green Connection Status Indicator** in the sidebar.
4. **Upload a student test video**.
5. Click **"Run Remote Evaluation 🚀"**.
6. **Watch the remote logs live** inside your SSH window to show the model execution on the server:
   ```bash
   kubectl logs -f -l app=attention --tail=20
   ```
7. Show the generated visual graphs, timeline curves, and table outputs inside the Streamlit UI.
