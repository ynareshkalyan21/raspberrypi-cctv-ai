# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 11/05/25
# this will run on raspberry pi 4B 2GB ram
import cv2
import threading
import time
import numpy as np
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import onnxruntime as ort
from datetime import datetime

# ---------- CONFIG ---------- #
RTSP_URL = 'rtsp://admin:admin12345@192.168.1.64/Streaming/Channels/101/'
FPS = 10
RECORD_INTERVAL = 30*10  # seconds

MODEL_PATH = 'yolov5s.onnx'

INFERENCE_MODEL_PATH = MODEL_PATH

# ---------- Globals ---------- #
latest_raw_frame = None        # Shared raw frame
latest_frame = None            # Processed (with inference) frame
frame_lock = threading.Lock()
lock = threading.Lock()

# ---------- Model Setup ---------- #
session = ort.InferenceSession(INFERENCE_MODEL_PATH)
input_name = session.get_inputs()[0].name

# ---------- YOLOv8 Helper Functions ---------- #
def preprocess(image):
    img = cv2.resize(image, (640, 640))
    img = img[..., ::-1].transpose(2, 0, 1).astype(np.float32)
    img /= 255.0
    return np.expand_dims(img, axis=0)

def postprocess(outputs, frame):
    pred = outputs[0][0]
    boxes, scores, class_ids = [], [], []
    h, w, _ = frame.shape
    for det in pred:
        conf = det[4]
        if conf < 0.5:
            continue
        class_id = np.argmax(det[5:])
        cx, cy, bw, bh = det[:4]
        x1 = int((cx - bw / 2) * w / 640)
        y1 = int((cy - bh / 2) * h / 640)
        x2 = int((cx + bw / 2) * w / 640)
        y2 = int((cy + bh / 2) * h / 640)
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        class_ids.append(class_id)
    return boxes, scores, class_ids

def draw_boxes(image, boxes, scores, class_ids):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        label = f"Class {class_id}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# ---------- Thread 1: RTSP Reader ---------- #
def rtsp_reader():
    global latest_raw_frame
    cap = cv2.VideoCapture(RTSP_URL)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            latest_raw_frame = frame.copy()

# ---------- Thread 2: Inference Loop ---------- #
def inference_loop():
    global latest_frame
    while True:
        with frame_lock:
            if latest_raw_frame is None:
                continue
            frame = latest_raw_frame.copy()
        start_time = time.time()
        inp = preprocess(frame)
        outputs = session.run(None, {input_name: inp})
        boxes, scores, class_ids = postprocess(outputs, frame)
        frame = draw_boxes(frame, boxes, scores, class_ids)
        duration = time.time() - start_time
        #add durarion to frame
        cv2.putText(frame, f"Inference Time: {duration:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        with lock:
            latest_frame = frame.copy()

        time.sleep(1.0 / FPS)

# ---------- Thread 3: Recording ---------- #
def record_stream():
    global latest_raw_frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    folder_path = None
    frame_count = 0
    last_rotation_time = time.time()

    while True:
        with frame_lock:
            if latest_raw_frame is None:
                continue
            frame = latest_raw_frame.copy()

        current_time = time.time()
        if out is None or (current_time - last_rotation_time) >= RECORD_INTERVAL:
            if out is not None:
                out.release()

            folder_path = f"recordings/{datetime.now().strftime('%Y-%m-%d')}"
            os.makedirs(folder_path, exist_ok=True)
            file_name = f"{datetime.now().strftime('%H-%M-%S')}.mp4"
            out_path = os.path.join(folder_path, file_name)
            out = cv2.VideoWriter(out_path, fourcc, FPS, (frame.shape[1], frame.shape[0]))
            print(f"Recording to {out_path}")
            last_rotation_time = current_time
            frame_count = 0

        out.write(frame)
        frame_count += 1
        time.sleep(1.0 / FPS)

# ---------- API ---------- #
app = FastAPI()

def generate_frames():
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.get("/")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# ---------- Main ---------- #
if __name__ == "__main__":
    # Start threads
    threading.Thread(target=rtsp_reader, daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()
    threading.Thread(target=record_stream, daemon=True).start()

    # Start API
    uvicorn.run(app, host="0.0.0.0", port=8000)
