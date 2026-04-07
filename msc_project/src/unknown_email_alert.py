import os
import cv2
import time
import smtplib
import numpy as np
import face_recognition
from email.message import EmailMessage
from datetime import datetime
from picamera2 import Picamera2

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "../models/yolov8n_416_opset12.onnx"
KNOWN_FACES_DIR = "../known_faces"
ALERTS_DIR = "../outputs/alerts"

IMG_SIZE = 416
CONF_THRES = 0.25
IOU_THRES = 0.45
RESOLUTION = (640, 480)

# Email settings
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "shivpatel5644@gmail.com"
SENDER_PASSWORD = "zlesdmtmzvvhcafa"
RECIPIENT_EMAIL = "formtf143@@gmail.com"

# Alert cooldown (seconds)
ALERT_COOLDOWN = 60

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

os.makedirs(ALERTS_DIR, exist_ok=True)

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

def load_known_faces():
    known_encodings = []
    known_names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        path = os.path.join(KNOWN_FACES_DIR, filename)

        if not os.path.isfile(path):
            continue

        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])

    return known_encodings, known_names

def send_email_alert(image_path, detected_time):
    msg = EmailMessage()
    msg["Subject"] = "Unknown Person Detected"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg.set_content(f"An unknown person was detected at {detected_time}.")

    with open(image_path, "rb") as f:
        img_data = f.read()
        msg.add_attachment(
            img_data,
            maintype="image",
            subtype="jpeg",
            filename=os.path.basename(image_path)
        )

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.starttls()
        smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
        smtp.send_message(msg)

# -----------------------------
# Load known faces
# -----------------------------
print("Loading known faces...")
known_face_encodings, known_face_names = load_known_faces()
print(f"Loaded {len(known_face_names)} known face(s).")

# -----------------------------
# Load detection model
# -----------------------------
print("Loading YOLO model...")
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

last_alert_time = 0

print("Unknown person alert system started. Press q to quit.")

try:
    while True:
        start = time.perf_counter()

        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_h, frame_w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
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
        unknown_detected = False

        # Convert for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for i in keep:
            cls_id = class_ids[i]
            if 0 <= cls_id < len(CLASSES):
                class_name = CLASSES[cls_id]
            else:
                continue

            x, y, w, h = boxes[i]
            score = scores[i]

            if class_name != "person":
                continue

            person_detected = True

            # Face recognition on full frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"

                if known_face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    frame,
                    name,
                    (left, max(20, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                if name == "Unknown":
                    unknown_detected = True

            # Person box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(
                frame,
                f"person {score:.2f}",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

        # Send alert if needed
        current_time = time.time()
        if unknown_detected and (current_time - last_alert_time > ALERT_COOLDOWN):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(ALERTS_DIR, f"unknown_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)

            detected_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            try:
                send_email_alert(image_path, detected_time)
                print(f"Alert sent: {image_path}")
                last_alert_time = current_time
            except Exception as e:
                print("Failed to send email:", e)

        end = time.perf_counter()
        fps = 1.0 / max(end - start, 1e-6)

        status_text = "UNKNOWN ALERT" if unknown_detected else ("KNOWN PERSON" if person_detected else "NO PERSON")

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status_text}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Unknown Person Email Alert", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Unknown person alert system stopped.")
