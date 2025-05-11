import cv2, time, os
from app.config import RECORD_DIR, FPS, RECORD_INTERVAL
from datetime import datetime
from app.streamer import latest_frame

def record_loop():
    out = None
    last_time = time.time()

    while True:
        if latest_frame is None:
            time.sleep(0.1)
            continue

        if out is None or time.time() - last_time > RECORD_INTERVAL:
            if out: out.release()
            folder = os.path.join(RECORD_DIR, datetime.now().strftime("%Y-%m-%d"))
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, f"{datetime.now().strftime('%H-%M-%S')}.mp4")
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (latest_frame.shape[1], latest_frame.shape[0]))
            last_time = time.time()

        out.write(latest_frame)
        time.sleep(1.0 / FPS)