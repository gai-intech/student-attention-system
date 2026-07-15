FROM python:3.11-slim

# Install system packages required by OpenCV, FFmpeg, and PyTorch/Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy API-serving dependencies and install them
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, deployment endpoints, and model weights
COPY src/ ./src/
COPY deployment/ ./deployment/
COPY models/ ./models/

EXPOSE 8000

# Start the FastAPI server on port 8000
CMD ["uvicorn", "deployment.main:app", "--host", "0.0.0.0", "--port", "8000"]
