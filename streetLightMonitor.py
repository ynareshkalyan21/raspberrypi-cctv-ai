# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 29/05/25
import cv2
import threading
import time
import numpy as np
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
import uvicorn
import onnxruntime as ort
from datetime import datetime
from pydantic import BaseModel # Added for FastAPI request body validation

# ---------- CONFIG ---------- #
# Ensure config.py exists in the same directory and contains these variables
from config import RECORD_DIR, RTSP_URL, FPS, RECORD_INTERVAL, MODEL_PATH
print(cv2.getBuildInformation())

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
INFERENCE_MODEL_PATH = MODEL_PATH # Path to your YOLOv8 ONNX model

# ---------- Globals ---------- #
latest_raw_frame = None        # Shared raw frame from RTSP reader
latest_frame = None            # Processed (with inference and ROI drawing) frame
frame_lock = threading.Lock()  # Lock for latest_raw_frame
inference_lock = threading.Lock() # Lock for latest_frame

# Street Light Monitoring Globals
street_light_roi = None  # Stores (x1, y1, x2, y2) coordinates of the street light ROI
street_light_state = "UNKNOWN" # Current detected state: "ON", "OFF", or "UNKNOWN"
previous_brightness = -1.0 # Stores the average brightness of the ROI from the previous frame for comparison

# Thresholds for state change detection (empirical values, may need tuning)
BRIGHTNESS_THRESHOLD_ON = 80.0  # If brightness is consistently above this, light is considered ON
BRIGHTNESS_THRESHOLD_OFF = 50.0 # If brightness is consistently below this, light is considered OFF
# Note: BRIGHTNESS_THRESHOLD_ON should be > BRIGHTNESS_THRESHOLD_OFF to create hysteresis
# This prevents rapid flickering between ON/OFF states due to minor brightness fluctuations.

DEBOUNCE_FRAMES = 5 # Number of consistent frames required for a state change to be confirmed
street_light_debounce_count = 0 # Counter for consistent state changes
last_notified_state = "UNKNOWN" # Tracks the last state for which a notification was sent, to avoid spamming

# ---------- Model Setup ---------- #
try:
    session = ort.InferenceSession(INFERENCE_MODEL_PATH, providers=["CPUExecutionProvider"], sess_options=sess_options)
    input_name = session.get_inputs()[0].name
    print(f"✅ ONNX model loaded from: {INFERENCE_MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading ONNX model from {INFERENCE_MODEL_PATH}: {e}")
    print("Please ensure 'yolov8n.onnx' is in the same directory or MODEL_PATH is correct in config.py.")
    # Exit or handle gracefully if model is essential
    exit()

# ---------- YOLOv8 Helper Functions ---------- #
def preprocess(image):
    """Preprocesses the image for YOLOv8 inference."""
    img = cv2.resize(image, (640, 640))
    img = img[..., ::-1].transpose(2, 0, 1).astype(np.float32) # BGR to RGB, HWC to CHW
    img /= 255.0 # Normalize to [0, 1]
    return np.expand_dims(img, axis=0) # Add batch dimension

def postprocess(outputs, frame):
    """Postprocesses YOLOv8 outputs to get bounding boxes, scores, and class IDs."""
    pred = outputs[0][0] # Assuming output format is (1, N, 84) for N detections
    boxes, scores, class_ids = [], [], []
    h, w, _ = frame.shape
    for det in pred:
        conf = det[4] # Confidence score
        if conf < 0.5: # Confidence threshold
            continue
        class_id = np.argmax(det[5:]) # Class ID (assuming 80 classes for COCO)
        cx, cy, bw, bh = det[:4] # Center x, center y, box width, box height
        # Convert normalized coordinates to pixel coordinates
        x1 = int((cx - bw / 2) * w / 640)
        y1 = int((cy - bh / 2) * h / 640)
        x2 = int((cx + bw / 2) * w / 640)
        y2 = int((cy + bh / 2) * h / 640)
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        class_ids.append(class_id)
    return boxes, scores, class_ids

def draw_boxes(image, boxes, scores, class_ids):
    """Draws bounding boxes, labels, and scores on the image."""
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        label = f"Class {class_id}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green rectangle
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Green text
    return image

# ---------- Thread 1: RTSP Reader ---------- #
def rtsp_reader():
    """Reads frames from the RTSP stream (or local video file) and updates latest_raw_frame."""
    global latest_raw_frame
    # Using a local video file for demonstration. Replace with RTSP_URL for live stream.
    # cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print(f"❌ Unable to open video source: {RTSP_URL} or local file.")
        return # Exit thread if cannot open source

    print("✅ Successfully opened video source.")
    print(f"Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Frame count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            print("❌ Failed to read frame from video stream. Attempting to reconnect...")
            cap.release()
            # Re-initialize capture in case of stream drop
            # cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap = cv2.VideoCapture("/Users/yarramsettinaresh/Downloads/tractor_basta.mp4", cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print("❌ Reconnection failed. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            else:
                print("✅ Reconnection successful.")
                continue

        with frame_lock: # Protect shared resource
            latest_raw_frame = frame_raw.copy()
        time.sleep(1.0 / FPS) # Control frame reading rate

# ---------- Thread 2: Inference Loop and Street Light Detection ---------- #
def inference_loop():
    """Performs object detection inference and street light monitoring."""
    global latest_frame, street_light_state, previous_brightness, street_light_debounce_count, last_notified_state

    while True:
        frame_to_process = None
        with frame_lock:
            if latest_raw_frame is not None:
                frame_to_process = latest_raw_frame.copy()

        if frame_to_process is None:
            # print(f"** No raw frame available for inference, skipping...**")
            time.sleep(0.1) # Wait a bit before checking again
            continue

        # --- Object Detection Inference ---
        start_time = time.time()
        inp = preprocess(frame_to_process)
        outputs = session.run(None, {input_name: inp})
        boxes, scores, class_ids = postprocess(outputs, frame_to_process)
        frame_processed = draw_boxes(frame_to_process, boxes, scores, class_ids)
        duration = time.time() - start_time
        cv2.putText(frame_processed, f"Inference Time: {duration:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- Street Light Monitoring ---
        if street_light_roi is not None:
            x1, y1, x2, y2 = street_light_roi
            # Ensure coordinates are within frame bounds
            h_frame, w_frame, _ = frame_processed.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_frame, x2)
            y2 = min(h_frame, y2)

            if x2 > x1 and y2 > y1: # Valid ROI (width and height > 0)
                roi = frame_processed[y1:y2, x1:x2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                current_brightness = np.mean(gray_roi)

                # Draw ROI rectangle and brightness on frame
                cv2.rectangle(frame_processed, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue rectangle for ROI
                cv2.putText(frame_processed, f"ROI Brightness: {current_brightness:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                new_state = street_light_state # Assume current state unless a change is confirmed

                if previous_brightness != -1.0: # Only compare after first valid brightness reading
                    # Check for transition from OFF/UNKNOWN to ON
                    if current_brightness > BRIGHTNESS_THRESHOLD_ON and street_light_state != "ON":
                        if street_light_debounce_count >= DEBOUNCE_FRAMES:
                            new_state = "ON"
                            street_light_debounce_count = 0
                        else:
                            street_light_debounce_count += 1
                    # Check for transition from ON/UNKNOWN to OFF
                    elif current_brightness < BRIGHTNESS_THRESHOLD_OFF and street_light_state != "OFF":
                        if street_light_debounce_count >= DEBOUNCE_FRAMES:
                            new_state = "OFF"
                            street_light_debounce_count = 0
                        else:
                            street_light_debounce_count += 1
                    else:
                        # If brightness is within the hysteresis range or consistent with current state, reset debounce
                        street_light_debounce_count = 0

                # Update state if a confirmed change occurred
                if new_state != street_light_state:
                    street_light_state = new_state
                    if street_light_state != last_notified_state: # Only notify if state actually changed from last notification
                        notification_message = f"Street Light is now: {street_light_state}!"
                        print(f"NOTIFICATION: {notification_message}")
                        # Simulate mobile notification by adding text to frame
                        cv2.putText(frame_processed, notification_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Red text for notification
                        last_notified_state = street_light_state
                else:
                    # If state hasn't changed, but ROI is defined, just update the text with current state
                    if street_light_state != "UNKNOWN":
                         cv2.putText(frame_processed, f"Light Status: {street_light_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow text for status

                previous_brightness = current_brightness # Store current brightness for next comparison
            else:
                cv2.putText(frame_processed, "Invalid ROI coordinates", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame_processed, "Set ROI for light monitoring", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # Cyan text

        with inference_lock: # Protect shared resource
            latest_frame = frame_processed.copy()
        time.sleep(1.0 / FPS) # Control inference rate

# ---------- Thread 3: Recording (Optional, currently commented out in main) ---------- #
def record_stream():
    """Records the processed video stream to files, rotating periodically."""
    global latest_frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
    out = None
    folder_path = None
    frame_count = 0
    last_rotation_time = time.time()

    while True:
        frame_to_record = None
        with inference_lock: # Get the latest processed frame
            if latest_frame is not None:
                frame_to_record = latest_frame.copy()

        if frame_to_record is None:
            time.sleep(0.1)
            continue

        current_time = time.time()
        # Rotate video file based on RECORD_INTERVAL or frame count
        if out is None or (current_time - last_rotation_time) >= RECORD_INTERVAL or frame_count >= (FPS * RECORD_INTERVAL):
            if out is not None:
                print(f"** Recording stopped, saving file: {out_path}")
                out.release() # Release the previous video writer

            folder_path = os.path.join(RECORD_DIR, datetime.now().strftime('%Y-%m-%d'))
            os.makedirs(folder_path, exist_ok=True) # Create daily folder if it doesn't exist
            file_name = f"{datetime.now().strftime('%H-%M-%S')}.mp4"
            out_path = os.path.join(folder_path, file_name)

            # Initialize new video writer
            # Use frame_to_record.shape[1] for width and frame_to_record.shape[0] for height
            out = cv2.VideoWriter(out_path, fourcc, FPS, (frame_to_record.shape[1], frame_to_record.shape[0]))
            if not out.isOpened():
                print(f"❌ Error: Could not open video writer for {out_path}")
                out = None # Reset out to prevent further write attempts to a bad file
                time.sleep(5) # Wait before retrying
                continue
            print(f"Recording to {out_path}")
            last_rotation_time = current_time
            frame_count = 0

        out.write(frame_to_record)
        frame_count += 1
        time.sleep(1.0 / FPS) # Control recording rate

# ---------- API ---------- #
app = FastAPI()

# Pydantic model for validating incoming ROI coordinates
class ROICoordinates(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

def generate_frames():
    """Generator function for streaming video frames via HTTP."""
    while True:
        frame_to_stream = None
        with inference_lock:
            if latest_frame is not None:
                frame_to_stream = latest_frame.copy()

        if frame_to_stream is None:
            # print("/live::generate_frames::No frame available") # Too verbose, removed
            time.sleep(0.1) # Wait a bit before checking again
            continue

        ret, jpeg = cv2.imencode('.jpg', frame_to_stream)
        if not ret:
            print("/live::generate_frames::Failed to encode frame")
            continue
        # print(f"/stream::generate_frames::Frame sent") # Too verbose, removed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serves the main HTML page."""
    # Ensure 'templates' directory exists and 'index.html' is inside it
    try:
        with open("templates/streetLight.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found in 'templates' directory.</h1>", status_code=404)

@app.get("/live")
async def video_feed():
    """Streams the live video feed with inference results."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/set_roi")
async def set_roi(coords: ROICoordinates):
    """API endpoint to set the Region of Interest (ROI) for street light monitoring."""
    global street_light_roi, street_light_state, previous_brightness, street_light_debounce_count, last_notified_state
    street_light_roi = (coords.x1, coords.y1, coords.x2, coords.y2)
    # Reset state and brightness when ROI is set/changed to re-evaluate
    street_light_state = "UNKNOWN"
    previous_brightness = -1.0
    street_light_debounce_count = 0
    last_notified_state = "UNKNOWN"
    print(f"Street light ROI set to: {street_light_roi}")
    return {"message": "ROI coordinates updated successfully", "roi": street_light_roi}

@app.get("/get_light_status")
async def get_light_status():
    """API endpoint to get the current street light status and its ROI."""
    global street_light_state, street_light_roi
    return {"status": street_light_state, "roi": street_light_roi}

@app.get("/recordings")
async def list_recordings_api():
    """API endpoint to list available video recordings."""
    all_files = []
    # os.walk traverses directory tree
    for root, _, files in os.walk(RECORD_DIR):
        for f in files:
            if f.endswith('.mp4'):
                # Store full path for FileResponse
                all_files.append(os.path.join(root, f))
    return sorted(all_files) # Return sorted list for consistent order

@app.get("/video")
async def get_video_file(path: str):
    """API endpoint to serve a specific recorded video file."""
    if os.path.exists(path) and path.startswith(RECORD_DIR):
        return FileResponse(path, media_type="video/mp4")
    return HTMLResponse(content="<h1>Error: Video not found or unauthorized path.</h1>", status_code=404)

# ---------- Main Execution Block ---------- #
if __name__ == "__main__":
    # Create recordings directory if it doesn't exist
    os.makedirs(RECORD_DIR, exist_ok=True)
    os.makedirs("templates", exist_ok=True) # Ensure templates directory exists

    # Start threads for video processing
    threading.Thread(target=rtsp_reader, daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()
    # Uncomment the line below to enable video recording
    # threading.Thread(target=record_stream, daemon=True).start()

    # Start FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)
