import os
from fastapi.responses import FileResponse
from app.config import RECORD_DIR

def list_recordings():
    all_files = []
    for root, _, files in os.walk(RECORD_DIR):
        for f in files:
            if f.endswith('.mp4'):
                all_files.append(os.path.join(root, f))
    return sorted(all_files)

def get_video(path: str):
    return FileResponse(path)