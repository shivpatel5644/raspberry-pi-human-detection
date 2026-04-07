import time
import csv
from datetime import datetime
import os

import cv2
import numpy as np
import psutil
from picamera2 import Picamera2
from gpiozero import LED


# -------------------
# LED Setup
# -------------------
green_led = LED(17)
yellow_led = LED(27)
red_led = LED(22)


# -------------------
# Settings
# -------------------
MODEL_PATH = "../models/yolov8n_416_opset12.onnx"
CONF_THRES = 0.10
IOU_THRES = 0.45
IMG_SIZE = 416
RESOLUTION = (320, 240)
RUN_SECONDS = 60

LOG_FILE = f"logs/pi_onnx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


# -------------------
# Create folders
# -------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# -------------------
# Load model
# -------------------
net = cv2.dnn.readNetFromONNX(MODEL_PATH)


# -------------------
# Camera setup
# -------------------
picam2 = Picamera2()
cfg = picam2.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"})
picam2.configure(cfg)
picam2.start()

time.sleep(2)


# -------------------
# Performance monitor
# -------------------
proc = psutil.Process()
proc.cpu_percent(interval=None)


# -------------------
# Logging file
# -------------------
with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["timestamp","fps","latency_ms","cpu_percent","ram_mb","detections","decision"])


print("Running YOLO detection...")
print("Logging to:", LOG_FILE)


# -------------------
# Video writer
# -------------------
video_path = f"outputs/pi_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(video_path, fourcc, 3.0, (RESOLUTION[0], RESOLUTION[1]))


# -------------------
# Timer
# -------------------
t0 = time.time()
frames = 0


# -------------------
# Main Loop
# -------------------
while time.time() - t0 < RUN_SECONDS:

    loop_start = time.perf_counter()

    frame = picam2.capture_array()
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    blob = cv2.dnn.blobFromImage(
        img,
        1/255.0,
        (IMG_SIZE, IMG_SIZE),
        swapRB=True,
        crop=False
    )

    infer_start = time.perf_counter()

    net.setInput(blob)
    out = net.forward()

    infer_end = time.perf_counter()

    latency_ms = (infer_end - infer_start) * 1000


    out = np.squeeze(out)

    if out.shape[0] == 84:
        preds = out.T
    else:
        preds = out


    boxes = []
    scores = []
    class_ids = []


    h, w = img.shape[:2]


    for p in preds:

        cx, cy, bw, bh = p[0], p[1], p[2], p[3]

        class_scores = p[4:]
        cls_id = int(np.argmax(class_scores))
        score = float(class_scores[cls_id])

        if score < CONF_THRES:
            continue


        x = int(cx - bw / 2)
        y = int(cy - bh / 2)
        bw = int(bw)
        bh = int(bh)


        boxes.append([x, y, bw, bh])
        scores.append(score)
        class_ids.append(cls_id)


    idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRES, IOU_THRES)


    keep = []
    if len(idxs) > 0:
        keep = idxs.flatten().tolist()


    det_count = len(keep)


    # -------------------
    # Decision Logic
    # -------------------
    decision = "IDLE"

    if det_count > 0:
        decision = "FOLLOW"

    if det_count >= 3:
        decision = "AVOID"


    # -------------------
    # LED Control
    # -------------------
    green_led.off()
    yellow_led.off()
    red_led.off()

    if decision == "FOLLOW":
        green_led.on()

    elif decision == "AVOID":
        red_led.on()

    else:
        yellow_led.on()


    # -------------------
    # Draw boxes
    # -------------------
    for i in keep:

        x, y, bw, bh = boxes[i]

        cv2.rectangle(
            img,
            (x, y),
            (x + bw, y + bh),
            (0,255,0),
            2
        )


    # -------------------
    # Performance
    # -------------------
    loop_end = time.perf_counter()

    fps = 1 / (loop_end - loop_start)

    frames += 1


    cpu = proc.cpu_percent(interval=None)
    ram = proc.memory_info().rss / (1024*1024)


    # -------------------
    # Logging
    # -------------------
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:

        w = csv.writer(f)

        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            f"{fps:.2f}",
            f"{latency_ms:.2f}",
            f"{cpu:.2f}",
            f"{ram:.2f}",
            det_count,
            decision
        ])


    # -------------------
    # Overlay text
    # -------------------
    cv2.putText(
        img,
        f"FPS: {fps:.2f}",
        (10,20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        img,
        f"Latency: {latency_ms:.0f} ms",
        (10,45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        img,
        decision,
        (10,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )


    writer.write(img)


writer.release()

print("Done. Frames processed:", frames)
print("Video saved:", video_path)
