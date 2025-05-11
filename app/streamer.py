import cv2, threading
from app.config import RTSP_URL

latest_frame = None

def rtsp_reader():
    global latest_frame
    cap = cv2.VideoCapture(RTSP_URL)
    while True:
        ret, frame = cap.read()
        if not ret: continue
        latest_frame = frame