import time
import cv2
import numpy as np
from picamera2 import Picamera2
from gpiozero import LED

# -------------------
# LED Setup
# -------------------
green_led = LED(17)   # FOLLOW
yellow_led = LED(27)  # IDLE
red_led = LED(22)     # AVOID

# -------------------
# Settings
# -------------------
MODEL_PATH = "yolov8n_416_opset12.onnx"
CONF_THRES = 0.10
IOU_THRES = 0.45
IMG_SIZE = 320
RESOLUTION = (256, 192)

# -------------------
# Load model
# -------------------
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

# -------------------
# Camera setup
# -------------------
picam2 = Picamera2()
cfg = picam2.create_video_configuration(
    main={"size": RESOLUTION, "format": "RGB888"}
)
picam2.configure(cfg)
picam2.start()
time.sleep(2)

# -------------------
# Helper: NMS
# -------------------
def get_nms_indices(boxes, scores, conf_thres, iou_thres):
    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    if len(idxs) == 0:
        return []
    return idxs.flatten().tolist()

# -------------------
# Main loop
# -------------------
print("Live demo started. Press 'q' to quit.")

while True:
    frame_skip = 0

    frame_skip += 1

    if frame_skip %2 != 0:
        display = cv2.resize(img, (640,480))
        cv2.imshow("YOLO Live Demo", display)
        cv2.waitKey(1)
        continue
    loop_start = time.perf_counter()

    frame = picam2.capture_array()
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Preprocess
    blob = cv2.dnn.blobFromImage(
        img,
        1 / 255.0,
        (IMG_SIZE, IMG_SIZE),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    out = net.forward()

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

    keep = get_nms_indices(boxes, scores, CONF_THRES, IOU_THRES)
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
    # Draw bounding boxes
    # -------------------
    for i in keep:
        x, y, bw, bh = boxes[i]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    loop_end = time.perf_counter()
    fps = 1 / max(loop_end - loop_start, 1e-6)

    # -------------------
    # Overlay
    # -------------------
    cv2.putText(
        img,
        f"FPS: {fps:.2f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        f"Objects: {det_count}",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        f"Decision: {decision}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # Bigger display for visibility
    display = cv2.resize(img, (640, 480))
    cv2.imshow("YOLO Live Demo", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# -------------------
# Cleanup
# -------------------
green_led.off()
yellow_led.off()
red_led.off()
picam2.stop()
cv2.destroyAllWindows()

print("Live demo stopped.")
