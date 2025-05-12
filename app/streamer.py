import cv2
from app.config import RTSP_URL

latest_frame = None

def rtsp_reader():
    print(f"starting RTSP reader,RTSP_URL:  {RTSP_URL}")
    global latest_frame
    cap = cv2.VideoCapture(RTSP_URL)
    print("RTSP reader started")
    while True:
        ret, frame = cap.read()
        if not ret: continue
        latest_frame = frame