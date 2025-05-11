from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
from app.streamer import latest_frame
from app.playback import list_recordings, get_video

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index():
    return open("templates/index.html").read()

@app.get("/live")
def live():
    def generate():
        while True:
            if latest_frame is None: continue
            _, jpeg = cv2.imencode('.jpg', latest_frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/recordings")
def recordings():
    return list_recordings()

@app.get("/video")
def video(path: str):
    return get_video(path)