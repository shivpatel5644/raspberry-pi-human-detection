import time
import csv
from datetime import datetime

import psutil
from ultralytics import YOLO
from picamera2 import Picamera2

# ---------- Settings ----------
CONF = 0.5
MODEL_NAME = "yolov8n.pt"   # auto-downloads first time
DURATION_SEC = 30           # run time for one experiment
RESOLUTION = (640, 480)

LOG_FILE = f"logs/pi_yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# ---------- Setup ----------
import os
os.makedirs("logs", exist_ok=True)

model = YOLO(MODEL_NAME)

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1)  # warmup

process = psutil.Process()
process.cpu_percent(interval=None)

with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "fps", "latency_ms", "cpu_proc_percent", "ram_mb", "num_detections"])

print("Running YOLO on Pi...")
print("Logging to:", LOG_FILE)

# ---------- Loop ----------
t_start = time.time()
frames = 0

while time.time() - t_start < DURATION_SEC:
    loop_start = time.perf_counter()

    frame = picam2.capture_array()

    infer_start = time.perf_counter()
    results = model.predict(source=frame, conf=CONF, verbose=False)
    infer_end = time.perf_counter()

    latency_ms = (infer_end - infer_start) * 1000.0

    det_count = 0
    if results and results[0].boxes is not None:
        det_count = len(results[0].boxes)

    loop_end = time.perf_counter()
    fps = 1.0 / (loop_end - loop_start)
    frames += 1

    cpu_proc = process.cpu_percent(interval=None)
    ram_mb = process.memory_info().rss / (1024 * 1024)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            f"{fps:.3f}",
            f"{latency_ms:.3f}",
            f"{cpu_proc:.2f}",
            f"{ram_mb:.2f}",
            det_count
        ])

picam2.stop()
print("Done. Frames processed:", frames)
