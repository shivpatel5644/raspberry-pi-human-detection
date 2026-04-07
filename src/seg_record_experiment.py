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

MODEL_PATH = "../models/yolov8n_seg_416_opset12.onnx"
OUTPUT_DIR = "../outputs"
LOG_DIR = "../logs"

IMG_SIZE = 416
CONF_THRES = 0.30
IOU_THRES = 0.45
RESOLUTION = (320, 240)

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

if GPIO_AVAILABLE:
    green_led = LED(17)   # FOLLOW
    yellow_led = LED(27)  # IDLE
    red_led = LED(22)     # AVOID

    def set_leds(state):
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
    def set_leds(state):
        pass

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def get_nms_indices(boxes, scores, conf_thres, iou_thres):
    idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    if len(idxs) == 0:
        return []
    return idxs.flatten().tolist()

# Create folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = os.path.join(OUTPUT_DIR, f"seg_experiment_{timestamp}.mp4")
log_path = os.path.join(LOG_DIR, f"seg_experiment_{timestamp}.csv")

print("Loading segmentation model...")
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
print("Model loaded.")

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": RESOLUTION, "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

time.sleep(2)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, RESOLUTION)

if not video_writer.isOpened():
    picam2.stop()
    raise RuntimeError("Failed to open video writer")

print("Segmentation recording started. Press q to quit.")

try:
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "fps",
            "object_count",
            "person_detected",
            "obstacle_detected",
            "decision"
        ])

        while True:
            start = time.perf_counter()

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
            layer_names = net.getUnconnectedOutLayersNames()
            outputs = net.forward(layer_names)

            det_out = outputs[0]
            proto_out = outputs[1]

            det = np.squeeze(det_out)      # (116, N)
            proto = np.squeeze(proto_out)  # (32, 104, 104)

            det = det.T  # (N, 116)

            boxes = []
            scores = []
            class_ids = []
            mask_coeffs = []

            x_factor = frame_w / IMG_SIZE
            y_factor = frame_h / IMG_SIZE

            for row in det:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                class_scores = row[4:84]
                coeffs = row[84:116]

                class_id = int(np.argmax(class_scores))
                score = float(class_scores[class_id])

                if score < CONF_THRES:
                    continue

                x = int((cx - w / 2) * x_factor)
                y = int((cy - h / 2) * y_factor)
                bw = int(w * x_factor)
                bh = int(h * y_factor)

                x = max(0, x)
                y = max(0, y)
                bw = max(1, bw)
                bh = max(1, bh)

                boxes.append([x, y, bw, bh])
                scores.append(score)
                class_ids.append(class_id)
                mask_coeffs.append(coeffs)

            keep = get_nms_indices(boxes, scores, CONF_THRES, IOU_THRES)

            person_detected = False
            obstacle_detected = False

            for i in keep:
                class_id = class_ids[i]

                if 0 <= class_id < len(CLASSES):
                    class_name = CLASSES[class_id]
                else:
                    class_name = f"class_{class_id}"

                if class_name == "person":
                    person_detected = True
                elif class_name in OBSTACLE_CLASSES:
                    obstacle_detected = True

            if obstacle_detected:
                decision = "AVOID"
            elif person_detected:
                decision = "FOLLOW"
            else:
                decision = "IDLE"

            set_leds(decision)

            overlay = img.copy()

            for i in keep:
                x, y, bw, bh = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                coeffs = mask_coeffs[i]

                if 0 <= class_id < len(CLASSES):
                    class_name = CLASSES[class_id]
                else:
                    class_name = f"class_{class_id}"

                proto_flat = proto.reshape(32, -1)
                mask = np.dot(coeffs, proto_flat)
                mask = sigmoid(mask).reshape(104, 104)

                mask = cv2.resize(mask, (frame_w, frame_h))
                mask = (mask > 0.5).astype(np.uint8)

                cropped_mask = np.zeros_like(mask)
                x2 = min(frame_w, x + bw)
                y2 = min(frame_h, y + bh)
                cropped_mask[y:y2, x:x2] = mask[y:y2, x:x2]

                if class_name == "person":
                    color = (0, 255, 0)
                elif class_name in OBSTACLE_CLASSES:
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 0)

                color_mask = np.zeros_like(img, dtype=np.uint8)
                color_mask[:, :] = color
                overlay = np.where(
                    cropped_mask[:, :, None] == 1,
                    (0.6 * overlay + 0.4 * color_mask).astype(np.uint8),
                    overlay
                )

                cv2.rectangle(overlay, (x, y), (x2, y2), color, 2)
                label = f"{class_name} {score:.2f}"
                cv2.putText(
                    overlay,
                    label,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

            end = time.perf_counter()
            fps = 1.0 / max(end - start, 1e-6)

            cv2.putText(
                overlay,
                f"FPS: {fps:.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                overlay,
                f"Segmentation Objects: {len(keep)}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                overlay,
                f"Decision: {decision}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{fps:.2f}",
                len(keep),
                person_detected,
                obstacle_detected,
                decision
            ])
            f.flush()

            video_writer.write(overlay)
            cv2.imshow("Segmentation Recording", overlay)

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
