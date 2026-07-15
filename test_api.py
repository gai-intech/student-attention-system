import requests
import json
import sys
import os

# Set the public server IP address and port
SERVER_IP = "13.60.62.49"
PORT = "8000"
URL = f"http://{SERVER_IP}:{PORT}/predict"

def test_prediction(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        sys.exit(1)
        
    print(f"Uploading '{video_path}' to Student Attention API at {URL}...")
    print("Processing models (YOLO detection, ByteTrack, behavior classifier, emotion classifier)...")
    
    # Open the video file in binary read mode
    with open(video_path, 'rb') as f:
        files = {'file': (os.path.basename(video_path), f, 'video/mp4')}
        # You can adjust target FPS sampling rate here (e.g. fps=1 processes 1 frame per second of video)
        params = {'fps': 1}
        
        try:
            response = requests.post(URL, files=files, params=params, timeout=300)
            if response.status_code == 200:
                print("\n=== Processing Complete ===")
                # Print the formatted JSON response
                print(json.dumps(response.json(), indent=4))
            else:
                print(f"\nError {response.status_code}: {response.text}")
        except Exception as e:
            print(f"\nConnection failed: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_video.mp4>")
        print("Example: python test_api.py sample_video.mp4")
        sys.exit(1)
        
    test_prediction(sys.argv[1])
