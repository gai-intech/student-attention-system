from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import shutil
import os
import sys

# Add the src/ directory to the system path to allow importing pipeline.py
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline import AttentionPipeline

app = FastAPI(title="Student Attention System API", version="1.0")

# Initialize the pipeline engine
pipe = AttentionPipeline()

@app.get("/")
def root():
    return {
        "service": "Student Attention System",
        "endpoints": ["/", "/predict", "/health"],
        "models": ["detection", "behavior", "emotion"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), fps: int = 2):
    # Save the uploaded video file to a temporary location
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        # Run the sequential pipeline on the video file (draw=False since we only need the numbers)
        for _ in pipe.process_video(tmp.name, target_fps=fps, draw=False):
            pass
        
        report = pipe.report()
        
        # Strip out numpy image crops as they cannot be JSON serialized directly
        report.pop("crops", None)
        
        # Round the values in the class attention curve for cleaner output
        report["class_curve"] = [[round(t, 2), round(a, 3)] for t, a in report["class_curve"]]
        
        return JSONResponse(report)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Securely delete the temporary file after processing
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
