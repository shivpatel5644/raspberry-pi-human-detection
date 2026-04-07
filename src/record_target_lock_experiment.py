import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime
from picamera2 import Picamera2

try:
    from gpiozero import LED
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "../models/yolov8n_416_opset12.onnx"
OUTPUT_DIR = "../outputs"
LOG_DIR = "../logs"

CONF_THRES = 0.25
IOU_THRES = 0.45
IMG_SIZE = 416
RESOLUTION = (640, 480)

TOO_FAR_RATIO = 0.03
TOO_CLOSE_RATIO = 0.30
STEER_THRESHOLD = 0.15

MAX_LOST_FRAMES = 15
LOCK_DISTANCE_THRESHOLD = 120  # pixels

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

OBSTACLE_CLASSES = {
    "chair", "couch", "bottle", "potted plant", "bed",
    "dining table", "tv", "laptop", "bench", "toilet"
}

# -----------------------------
# LED setup
# -----------------------------
if GPIO_AVAILABLE:
    green_led = LED(17)   # movement
    yellow_led = LED(27)  # idle / hold
    red_led = LED(22)     # avoid

    def set_leds(state):
        green_led.off()
        yellow_led.off()
        red_led.off()

        if state == "AVOID":
            red_led.on()
        elif state in [
            "STEER LEFT", "STEER RIGHT",
            "STEER LEFT + FORWARD", "STEER RIGHT + FORWARD",
            "MOVE FORWARD", "MOVE BACKWARD"
        ]:
            green_led.on()
        else:
            yellow_led.on()
else:
    def set_leds(state):
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
    out = np.squeeze(out)

    if len(out.shape) != 2:
        return [], [], []

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

        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)

        boxes.append([x, y, w, h])
        scores.append(score)
        class_ids.append(cls_id)

    return boxes, scores, class_ids

def center_distance(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

# -----------------------------
# Prepare output folders
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = os.path.join(OUTPUT_DIR, f"target_lock_experiment_{timestamp}.mp4")
log_path = os.path.join(LOG_DIR, f"target_lock_experiment_{timestamp}.csv")

# -----------------------------
# Load model
# -----------------------------
print("Loading model...")
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
print("Model loaded.")

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

# -----------------------------
# Video writer
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, RESOLUTION)

if not video_writer.isOpened():
    picam2.stop()
    raise RuntimeError("Failed to open video writer")

# -----------------------------
# Target lock state
# -----------------------------
locked_target_center = None
lost_frames = 0

print("Target lock recording started. Press q to quit.")

try:
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "fps",
            "object_count",
            "obstacle_detected",
            "person_candidate_count",
            "target_locked",
            "lock_status",
            "lost_frames",
            "person_center_x",
            "frame_center_x",
            "distance_ratio",
            "steering_percent",
            "movement_command"
        ])

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

            obstacle_detected = False
            person_candidates = []

            for i in keep:
                cls_id = class_ids[i]

                if 0 <= cls_id < len(CLASSES):
                    class_name = CLASSES[cls_id]
                else:
                    class_name = f"class_{cls_id}"

                x, y, w, h = boxes[i]
                score = scores[i]
                center_x = x + w // 2
                center_y = y + h // 2
                area = w * h

                if class_name == "person":
                    person_candidates.append({
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "score": score,
                        "center": (center_x, center_y),
                        "area": area
                    })
                elif class_name in OBSTACLE_CLASSES:
                    obstacle_detected = True

            best_person = None
            lock_status = "UNLOCKED"

            # -----------------------------
            # Target selection + target lock
            # -----------------------------
            if person_candidates:
                if locked_target_center is None:
                    best_person = max(person_candidates, key=lambda p: p["area"])
                    locked_target_center = best_person["center"]
                    lost_frames = 0
                    lock_status = "LOCKED (NEW)"
                else:
                    best_person = min(
                        person_candidates,
                        key=lambda p: center_distance(p["center"], locked_target_center)
                    )

                    dist = center_distance(best_person["center"], locked_target_center)

                    if dist <= LOCK_DISTANCE_THRESHOLD:
                        locked_target_center = best_person["center"]
                        lost_frames = 0
                        lock_status = "LOCKED"
                    else:
                        lost_frames += 1
                        lock_status = f"SEARCHING ({lost_frames})"

                        if lost_frames > MAX_LOST_FRAMES:
                            best_person = max(person_candidates, key=lambda p: p["area"])
                            locked_target_center = best_person["center"]
                            lost_frames = 0
                            lock_status = "RE-LOCKED"
            else:
                lost_frames += 1
                lock_status = f"LOST ({lost_frames})"

                if lost_frames > MAX_LOST_FRAMES:
                    locked_target_center = None
                    best_person = None
                    lock_status = "UNLOCKED"

            # -----------------------------
            # Behaviour logic
            # -----------------------------
            left_boundary = frame_w // 3
            right_boundary = 2 * frame_w // 3
            frame_center_x = frame_w // 2

            command = "IDLE"
            steering_text = "0%"
            steering_percent = 0
            distance_ratio_text = "0.000"
            distance_ratio = 0.0
            person_center_x = -1
            target_locked = best_person is not None and locked_target_center is not None

            if obstacle_detected:
                command = "AVOID"

            elif best_person is not None:
                x = best_person["x"]
                y = best_person["y"]
                w = best_person["w"]
                h = best_person["h"]

                person_center_x = x + (w // 2)

                person_area = w * h
                frame_area = frame_w * frame_h
                distance_ratio = person_area / frame_area
                distance_ratio_text = f"{distance_ratio:.3f}"

                error = person_center_x - frame_center_x
                steering = error / frame_center_x
                steering_percent = int(abs(steering) * 100)
                steering_text = f"{steering_percent}%"

                if distance_ratio > TOO_CLOSE_RATIO:
                    command = "MOVE BACKWARD"

                elif distance_ratio < TOO_FAR_RATIO:
                    if steering < -STEER_THRESHOLD:
                        command = "STEER LEFT + FORWARD"
                    elif steering > STEER_THRESHOLD:
                        command = "STEER RIGHT + FORWARD"
                    else:
                        command = "MOVE FORWARD"

                else:
                    if steering < -STEER_THRESHOLD:
                        command = "STEER LEFT"
                    elif steering > STEER_THRESHOLD:
                        command = "STEER RIGHT"
                    else:
                        command = "HOLD POSITION"

            else:
                command = "IDLE"

            set_leds(command)

            # -----------------------------
            # Draw guide lines
            # -----------------------------
            cv2.line(img, (left_boundary, 0), (left_boundary, frame_h), (255, 255, 255), 1)
            cv2.line(img, (frame_center_x, 0), (frame_center_x, frame_h), (0, 255, 255), 1)
            cv2.line(img, (right_boundary, 0), (right_boundary, frame_h), (255, 255, 255), 1)

            # -----------------------------
            # Draw detections
            # -----------------------------
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

            # Highlight locked person
            if best_person is not None:
                x = best_person["x"]
                y = best_person["y"]
                w = best_person["w"]
                h = best_person["h"]
                center_x, center_y = best_person["center"]

                cv2.circle(img, (center_x, center_y), 6, (0, 255, 0), -1)
                cv2.putText(
                    img,
                    "TARGET LOCK",
                    (x, max(40, y - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
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
                f"Command: {command}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"Steering: {steering_text}",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                img,
                f"Distance ratio: {distance_ratio_text}",
                (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 255, 200),
                2
            )

            cv2.putText(
                img,
                f"Lock status: {lock_status}",
                (10, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 200, 200),
                2
            )

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{fps:.2f}",
                len(keep),
                obstacle_detected,
                len(person_candidates),
                target_locked,
                lock_status,
                lost_frames,
                person_center_x,
                frame_center_x,
                f"{distance_ratio:.4f}",
                steering_percent,
                command
            ])
            f.flush()

            video_writer.write(img)
            cv2.imshow("Target Lock Recording", img)

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
    video_writer.release()
    cv2.destroyAllWindows()

print("Video saved:", video_path)
print("Log saved:", log_path)
