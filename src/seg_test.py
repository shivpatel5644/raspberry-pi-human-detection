import cv2
import numpy as np
import time
from picamera2 import Picamera2

MODEL_PATH = "../models/yolov8n_seg_416_opset12.onnx"
IMG_SIZE = 416

print("Loading segmentation model...")
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640,480), "format":"RGB888"})
picam2.configure(config)
picam2.start()

time.sleep(2)

print("Segmentation model test started. Press q to quit.")

while True:

    start = time.time()

    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1/255,
        size=(IMG_SIZE, IMG_SIZE),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)

    outputs = net.forward()

    end = time.time()
    fps = 1/(end-start)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

    cv2.imshow("Segmentation Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
