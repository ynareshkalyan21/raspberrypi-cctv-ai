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
from config import RECORD_DIR, RTSP_URL,FPS,RECORD_INTERVAL,MODEL_PATH

import subprocess

def speak(text, lang='te'):
    # Use espeak with language code (te for Telugu)
    subprocess.call(['espeak', '-v', lang, text])

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
INFERENCE_MODEL_PATH = MODEL_PATH

# ---------- Globals ---------- #
latest_raw_frame = None  # Shared raw frame from RTSP reader
latest_frame = None  # Processed (with inference and light detection) frame
frame_lock = threading.Lock()  # Lock for latest_raw_frame
inference_lock = threading.Lock()  # Lock for latest_frame

# ---------- Street Light Monitoring Globals ---------- #
# Define the coordinates for the street light ROI (x1, y1, x2, y2)
# IMPORTANT: Adjust these coordinates to match the location of your street light in the video feed.
STREET_LIGHT_ROI = (610, 270, 720, 370)    # Example: top-left (500,100), bottom-right (600,200)
STREET_LIGHT_OFF_AT = None
street_light_status = "UNKNOWN"  # Current status: "ON", "OFF", "UNKNOWN"
last_light_intensity = -1  # Stores the last calculated intensity of the ROI
# Thresholds for detecting ON/OFF state. These will likely need calibration.
# light_off_threshold: If intensity drops below this, start counting for OFF.
light_off_threshold = 170  # Example: Adjust based on your light's 'off' brightness
# light_on_threshold: If intensity rises above this, start counting for ON.
light_on_threshold = 170  # Example: Adjust based on your light's 'on' brightness

consecutive_off_frames = 0  # Counter for consecutive frames below off_threshold
consecutive_on_frames = 0  # Counter for consecutive frames above on_threshold
REQUIRED_CONSECUTIVE_FRAMES = 5  # Number of consecutive frames needed to confirm a state change

# ---------- Model Setup ---------- #
try:
    session = ort.InferenceSession(INFERENCE_MODEL_PATH, providers=["CPUExecutionProvider"], sess_options=sess_options)
    input_name = session.get_inputs()[0].name
    print(f"ONNX model loaded successfully from {INFERENCE_MODEL_PATH}")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    print("Please ensure MODEL_PATH is correct and the ONNX model is valid.")
    session = None  # Set session to None to handle cases where model loading fails


# ---------- YOLOv8 Helper Functions ---------- #
def preprocess(image):
    """
    Preprocesses the image for YOLOv8 inference.
    Resizes, converts color, transposes, normalizes, and adds batch dimension.
    """
    img = cv2.resize(image, (640, 640))
    img = img[..., ::-1].transpose(2, 0, 1).astype(np.float32)  # BGR to RGB, HWC to CHW
    img /= 255.0  # Normalize to [0, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension


def postprocess(outputs, frame):
    """
    Postprocesses the YOLOv8 model outputs to extract bounding boxes, scores, and class IDs.
    Applies non-max suppression if needed (though not explicitly implemented here,
    YOLOv8 models often have NMS built-in or applied in a separate step).
    """
    pred = outputs[0][0]  # Assuming output format is (1, N, 6) where N is num_detections
    boxes, scores, class_ids = [], [], []
    h, w, _ = frame.shape
    for det in pred:
        conf = det[4]  # Confidence score for the object
        if conf < 0.5:  # Confidence threshold
            continue
        class_id = np.argmax(det[5:])  # Class ID is the index of the max probability
        cx, cy, bw, bh = det[:4]  # Center_x, Center_y, Box_width, Box_height

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
    """
    Draws bounding boxes, scores, and class labels on the image.
    """
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        label = f"Class {class_id}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


# ---------- Street Light Monitoring Function ---------- #
def monitor_street_light(frame, roi):
    """
    Monitors the intensity of the specified ROI to determine if the street light is ON or OFF.
    Updates global status variables and draws indicators on the frame.
    """
    global street_light_status, last_light_intensity, consecutive_off_frames, consecutive_on_frames, STREET_LIGHT_OFF_AT

    x1, y1, x2, y2 = roi

    # Ensure ROI is within frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    # Check for invalid ROI dimensions
    if x2 <= x1 or y2 <= y1:
        print("Invalid Street Light ROI coordinates. Please check STREET_LIGHT_ROI.")
        # Draw a red rectangle to indicate invalid ROI
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)
        cv2.putText(frame, "INVALID ROI", (roi[0], roi[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    roi_frame = frame[y1:y2, x1:x2]

    # Check if ROI is empty (e.g., due to out-of-bounds coordinates after clamping)
    if roi_frame.size == 0:
        print("Street Light ROI is empty after clamping. Check coordinates.")
        return frame

    # Convert ROI to grayscale and calculate average intensity
    gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    current_intensity = np.mean(gray_roi)

    # State machine for light detection
    if last_light_intensity == -1:  # Initial state
        last_light_intensity = current_intensity
        if current_intensity > light_on_threshold:
            street_light_status = "ON"
        else:
            street_light_status = "OFF"
            STREET_LIGHT_OFF_AT = datetime.now()  # Record the time when light turned OFF
        print(f"Initial Street Light Status: {street_light_status} (Intensity: {current_intensity:.2f})")
    else:
        # Check for light turning OFF
        if current_intensity < light_off_threshold:
            consecutive_off_frames += 1
            consecutive_on_frames = 0  # Reset ON counter
            if street_light_status != "OFF":
                street_light_status = "OFF"
                STREET_LIGHT_OFF_AT = datetime.now()  # Record the time when light turned OFF
                print(f"🚨 NOTIFICATION: Street Light turned OFF! (Intensity: {current_intensity:.2f})")
                speak("Corrent Vellipoyindi")
                time.sleep(5)
                # Here you would integrate with a mobile notification service
        # Check for light turning ON
        elif current_intensity > light_on_threshold:
            consecutive_on_frames += 1
            consecutive_off_frames = 0  # Reset OFF counter
            if street_light_status != "ON":
                street_light_status = "ON"

                print(f"💡 NOTIFICATION: Street Light turned ON! (Intensity: {current_intensity:.2f})")
                # Here you would integrate with a mobile notification service
        else:
            # Intensity is in an ambiguous zone or stable, reset counters
            consecutive_off_frames = 0
            consecutive_on_frames = 0
        # Print current status and intensity

        last_light_intensity = current_intensity  # Update last intensity for next comparison

    # Draw ROI and status on frame
    # Draw the ROI rectangle in blue
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # Put text indicating light status and current intensity
    cv2.putText(frame, f"Light: {street_light_status} ({current_intensity:.0f})", (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Blue text
    if street_light_status == "OFF":
        duration_in_min = (datetime.now() - STREET_LIGHT_OFF_AT).total_seconds() / 60
        duration_in_min = int(duration_in_min)

        # Draw a red circle if the light is OFF
        if (datetime.now() - STREET_LIGHT_OFF_AT).total_seconds() < 3*60:
            print(f"Street Light OFF for {duration_in_min} minutes")
            speak(f"{duration_in_min} , nimishalu {duration_in_min},  nimishalu ayyindi")
            speak(f"Corrent Vellipoyi")
            time.sleep(15)
        # draw minute duration_in_min text
        cv2.putText(frame, f"OFF for: {duration_in_min:.1f} min", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    return frame


# ---------- Thread 1: RTSP Reader ---------- #
def rtsp_reader():
    """
    Reads frames from the RTSP stream and updates the global latest_raw_frame.
    """
    global latest_raw_frame
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print(f"Failed to open RTSP stream at {RTSP_URL}. Retrying in 5 seconds...")
        time.sleep(5)
        # Attempt to reconnect if failed
        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            print(f"Still failed to open RTSP stream. Exiting RTSP reader thread.")
            return

    cc = 0
    while True:
        ret, frame_raw = cap.read()
        if not ret:
            print("❌ Failed to read frame from RTSP stream. Attempting to reconnect...")
            cap.release()
            time.sleep(2)  # Wait before attempting to reconnect
            cap = cv2.VideoCapture(RTSP_URL)
            if not cap.isOpened():
                print("Reconnection failed. Will retry...")
                time.sleep(5)  # Longer wait if reconnection fails
            continue

        with frame_lock:  # Protect access to latest_raw_frame
            if latest_raw_frame is None:
                print("Setting first raw frame.")
            latest_raw_frame = frame_raw.copy()
            cc += 1
            # print(f"RTSP Reader: {cc} frames read") # Uncomment for debugging frame rate


# ---------- Thread 2: Inference Loop ---------- #
def inference_loop():
    """
    Performs object detection and street light monitoring on frames,
    then updates the global latest_frame for streaming and recording.
    """
    global latest_frame
    ic = 0
    while True:
        frame_to_process = None
        with frame_lock:  # Acquire lock to read latest_raw_frame safely
            if latest_raw_frame is not None:
                frame_to_process = latest_raw_frame.copy()
            else:
                # print(f"** No raw frame available for inference, skipping...**")
                time.sleep(0.1)  # Small sleep to avoid busy-waiting
                continue

        if frame_to_process is None:
            continue

        start_time = time.time()

        # Perform YOLOv8 inference if model is loaded
        if session:
            inp = preprocess(frame_to_process)
            outputs = session.run(None, {input_name: inp})
            boxes, scores, class_ids = postprocess(outputs, frame_to_process)
            frame_to_process = draw_boxes(frame_to_process, boxes, scores, class_ids)
        else:
            cv2.putText(frame_to_process, "Model not loaded!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # --- New: Monitor street light ---
        frame_to_process = monitor_street_light(frame_to_process, STREET_LIGHT_ROI)
        # ---------------------------------

        duration = time.time() - start_time
        # Add inference duration to frame
        cv2.putText(frame_to_process, f"Inference Time: {duration:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        ic += 1
        # print(f"ic:{ic} Inference Time: {duration:.2f}s") # Uncomment for debugging

        with inference_lock:  # Protect access to latest_frame
            latest_frame = frame_to_process.copy()


# ---------- Thread 3: Recording ---------- #
def record_stream():
    """
    Records the processed video stream to MP4 files, rotating files based on RECORD_INTERVAL.
    """
    global latest_frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = None
    out_path = None
    frame_count = 0

    os.makedirs(RECORD_DIR, exist_ok=True)  # Ensure recording directory exists

    while True:
        frame_to_record = None
        with inference_lock:  # Acquire lock to read latest_frame safely
            if latest_frame is not None:
                frame_to_record = latest_frame.copy()
            else:
                time.sleep(0.1)  # Small sleep to avoid busy-waiting
                continue

        if frame_to_record is None:
            continue

        # Initialize or rotate video file
        if out is None or frame_count >= RECORD_INTERVAL:
            if out is not None:
                print(f"** Recording stopped, saving file: {out_path}")
                out.release()  # Release the previous video writer

            # Create new folder for current date if it doesn't exist
            folder_path = os.path.join(RECORD_DIR, datetime.now().strftime('%Y-%m-%d'))
            os.makedirs(folder_path, exist_ok=True)

            # Generate new file name with timestamp
            file_name = f"{datetime.now().strftime('%H-%M-%S')}.mp4"
            out_path = os.path.join(folder_path, file_name)

            # Initialize new video writer
            # Ensure frame_to_record has valid dimensions before initializing VideoWriter
            if frame_to_record.shape[0] > 0 and frame_to_record.shape[1] > 0:
                out = cv2.VideoWriter(out_path, fourcc, FPS, (frame_to_record.shape[1], frame_to_record.shape[0]))
                print(f"Recording to {out_path}")
                frame_count = 0  # Reset frame count for the new file
            else:
                print("Warning: Frame has invalid dimensions for video writer. Skipping recording initialization.")
                out = None  # Prevent further write attempts until a valid frame comes
                time.sleep(1)  # Wait before retrying

        if out is not None:
            out.write(frame_to_record)
            frame_count += 1

        time.sleep(1.0 / FPS)  # Control recording frame rate


# ---------- API ---------- #
app = FastAPI()


def generate_frames():
    """
    Generator function to yield JPEG frames for the MJPEG streaming endpoint.
    """
    ccc = 0
    while True:
        frame_to_send = None
        with inference_lock:  # Acquire lock to read latest_frame safely
            if latest_frame is not None:
                frame_to_send = latest_frame.copy()
            else:
                # print("/live::generate_frames::No frame available")
                time.sleep(0.1)  # Small sleep to avoid busy-waiting
                continue

        if frame_to_send is None:
            continue

        ret, jpeg = cv2.imencode('.jpg', frame_to_send)
        if not ret:
            print("/live::generate_frames::Failed to encode frame")
            continue
        ccc += 1
        # print(f"/stream::generate_frames::{ccc} frames sent") # Uncomment for debugging
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


def list_recordings():
    """
    Lists all recorded MP4 files.
    """
    all_files = []
    # os.walk traverses directories recursively
    for root, _, files in os.walk(RECORD_DIR):
        for f in files:
            if f.endswith('.mp4'):
                # Append full path to the list
                all_files.append(os.path.join(root, f))
    return sorted(all_files, reverse=True)  # Sort by most recent first


@app.get("/", response_class=HTMLResponse)
async def index():
    """
    Serves the main HTML page for the video stream and recordings.
    """
    # Read the index.html file
    try:
        with open("templates/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: templates/index.html not found!</h1>", status_code=404)


@app.get("/live")
async def video_feed():
    """
    Endpoint for MJPEG streaming of the live video feed.
    """
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/recordings_list")
def recordings_list():
    """
    Endpoint to list all available video recordings.
    """
    return list_recordings()


@app.get("/video")
def video(path: str):
    """
    Endpoint to serve a specific video file.
    """
    # Basic security check to prevent directory traversal
    if not os.path.exists(path) or not path.startswith(RECORD_DIR):
        return {"error": "Invalid video path"}, 400
    return FileResponse(path)


@app.get("/street_light_status")
def get_street_light_status():
    """
    New endpoint to get the current status of the street light.
    """
    global street_light_status
    return {"status": street_light_status}


# ---------- Main ---------- #
if __name__ == "__main__":
    # Create the recordings directory if it doesn't exist
    os.makedirs(RECORD_DIR, exist_ok=True)

    # Start threads
    print("Starting RTSP reader thread...")
    threading.Thread(target=rtsp_reader, daemon=True).start()
    print("Starting inference loop thread...")
    threading.Thread(target=inference_loop, daemon=True).start()
    # print("Starting recording stream thread...")
    # threading.Thread(target=record_stream, daemon=True).start()

    # Start API
    print("Starting FastAPI application on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
