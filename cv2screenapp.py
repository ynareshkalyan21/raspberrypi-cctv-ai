# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 12/05/25
import cv2
import numpy as np
import onnxruntime as ort

# === RTSP URL of your Hikvision DVR ===
RTSP_URL = 'rtsp://admin:admin12345@192.168.1.33:554/Streaming/Channels/101/'

# === Load ONNX model with ONNXRuntime ===
session = ort.InferenceSession('/home/ai/Downloads/yolov5s.onnx', providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


# === Preprocessing function ===
def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1]  # BGR to RGB
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# === Postprocessing function ===
def postprocess(output, frame, conf_threshold=0.4):
    predictions = output[0]  # shape: (1, num_predictions, 85)
    predictions = np.squeeze(predictions, axis=0)  # shape: (num_predictions, 85)

    boxes, scores, class_ids = [], [], []
    image_height, image_width = frame.shape[:2]

    for row in predictions:
        confidence = row[4]
        if confidence > conf_threshold:
            class_scores = row[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            if class_confidence > conf_threshold:
                cx, cy, w, h = row[0:4]
                x1 = int((cx - w / 2) * image_width / 640)
                y1 = int((cy - h / 2) * image_height / 640)
                x2 = int((cx + w / 2) * image_width / 640)
                y2 = int((cy + h / 2) * image_height / 640)

                boxes.append((x1, y1, x2, y2))
                scores.append(float(class_confidence))
                class_ids.append(class_id)

    return boxes, scores, class_ids


# === Simple class labels (COCO 80 classes, add full list if needed) ===
LABELS = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
          "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
          "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant"]

# === Open video stream ===
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    print("❌ Failed to open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)

    outputs = session.run(None, {input_name: input_tensor})
    boxes, scores, class_ids = postprocess(np.array(outputs), frame)

    # Draw results
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        label = f"{LABELS[class_id]}: {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show live detection
    cv2.imshow("Hikvision Live View - YOLOv5", frame)
    if cv2.waitKey(1) == 27:
        break  # ESC to exit

cap.release()
cv2.destroyAllWindows()
