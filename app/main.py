import threading

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2

from app.recorder import record_loop
from app.streamer import latest_frame, rtsp_reader
from app.playback import list_recordings, get_video
import uvicorn
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

if __name__ == "__main__":
    # Start threads
    threading.Thread(target=rtsp_reader, daemon=True).start()
    # threading.Thread(target=inference_loop, daemon=True).start()
    threading.Thread(target=record_loop, daemon=True).start()

    # Start API
    uvicorn.run(app, host="0.0.0.0", port=8000)