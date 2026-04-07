import time
import cv2
import numpy as np
from picamera2 import Picamera2

# Optional LED support
try:
    from gpiozero import LED
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "../models/yolov8n_416_opset12.onnx"
CONF_THRES = 0.25
IOU_THRES = 0.45
IMG_SIZE = 416
RESOLUTION = (640, 480)

# COCO classes for YOLOv8
CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
    "hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table",
    "toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

# Classes to treat as obstacles for demo logic
OBSTACLE_CLASSES = {
    "chair", "couch", "bottle", "potted plant", "bed",
    "dining table", "tv", "laptop", "bench", "toilet"
}

# -----------------------------
# LED setup
# -----------------------------
if GPIO_AVAILABLE:
    green_led = LED(17)   # FOLLOW
    yellow_led = LED(27)  # IDLE
    red_led = LED(22)     # AVOID

    def set_leds(state: str) -> None:
        green_led.off()
        yellow_led.off()
        red_led.off()

        if state == "FOLLOW":
            green_led.on()
        elif state == "AVOID":
            red_led.on()
        else:
            yellow_led.on()
else:
    def set_leds(state: str) -> None:
        pass

# -----------------------------
# Helper functions
# -----------------------------
def get_nms_indices(boxes, scores, conf_thres, iou_thres):
    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    if len(idxs) == 0:
        return []
    return idxs.flatten().tolist()

def parse_yolo_output(out, frame_w, frame_h):
    """
    Parses YOLOv8 ONNX output from OpenCV DNN.
    Handles common output shapes such as:
    - (1, 84, N)
    - (84, N)
    - (N, 84)
    """
    out = np.squeeze(out)

    if len(out.shape) != 2:
        return [], [], []

    # Convert to shape (N, attributes)
    if out.shape[0] in [84, 85]:
        preds = out.T
    else:
        preds = out

    boxes = []
    scores = []
    class_ids = []

    x_factor = frame_w / IMG_SIZE
    y_factor = frame_h / IMG_SIZE

    for p in preds:
        if len(p) < 5:
            continue

        cx, cy, bw, bh = p[0], p[1], p[2], p[3]
        class_scores = p[4:]

        cls_id = int(np.argmax(class_scores))
        score = float(class_scores[cls_id])

        if score < CONF_THRES:
            continue

        x = int((cx - bw / 2) * x_factor)
        y = int((cy - bh / 2) * y_factor)
        w = int(bw * x_factor)
        h = int(bh * y_factor)

        # Clamp to image bounds
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)

        boxes.append([x, y, w, h])
        scores.append(score)
        class_ids.append(cls_id)

    return boxes, scores, class_ids

# -----------------------------
# Load model
# -----------------------------
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

# -----------------------------
# Camera setup
# -----------------------------
picam2 = Picamera2()
cfg = picam2.create_video_configuration(
    main={"size": RESOLUTION, "format": "RGB888"}
)
picam2.configure(cfg)
picam2.start()
time.sleep(2)

print("Live demo started. Press 'q' to quit.")

try:
    while True:
        loop_start = time.perf_counter()

        frame = picam2.capture_array()
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_h, frame_w = img.shape[:2]

        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1 / 255.0,
            size=(IMG_SIZE, IMG_SIZE),
            swapRB=True,
            crop=False
        )

        net.setInput(blob)
        out = net.forward()

        boxes, scores, class_ids = parse_yolo_output(out, frame_w, frame_h)
        keep = get_nms_indices(boxes, scores, CONF_THRES, IOU_THRES)

        person_detected = False
        obstacle_detected = False

        for i in keep:
            cls_id = class_ids[i]
            if 0 <= cls_id < len(CLASSES):
                class_name = CLASSES[cls_id]
            else:
                class_name = f"class_{cls_id}"

            if class_name == "person":
                person_detected = True
            elif class_name in OBSTACLE_CLASSES:
                obstacle_detected = True

        # Decision logic
        if obstacle_detected:
            decision = "AVOID"
        elif person_detected:
            decision = "FOLLOW"
        else:
            decision = "IDLE"

        set_leds(decision)

        # Draw detections
        for i in keep:
            x, y, w, h = boxes[i]
            cls_id = class_ids[i]
            score = scores[i]

            if 0 <= cls_id < len(CLASSES):
                class_name = CLASSES[cls_id]
            else:
                class_name = f"class_{cls_id}"

            if class_name == "person":
                color = (0, 255, 0)
            elif class_name in OBSTACLE_CLASSES:
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)

            label = f"{class_name} {score:.2f}"

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                img,
                label,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        loop_end = time.perf_counter()
        fps = 1.0 / max(loop_end - loop_start, 1e-6)

        cv2.putText(
            img,
            f"FPS: {fps:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            f"Objects: {len(keep)}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            f"Decision: {decision}",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("YOLOv8 Live Demo", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    if GPIO_AVAILABLE:
        green_led.off()
        yellow_led.off()
        red_led.off()

    picam2.stop()
    cv2.destroyAllWindows()
    print("Live demo stopped.")
