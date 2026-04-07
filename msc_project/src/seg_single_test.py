import cv2
import numpy as np
import time
from picamera2 import Picamera2

MODEL_PATH = "../models/yolov8n_seg_416_opset12.onnx"
IMG_SIZE = 416

print("Loading segmentation model...")
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
print("Model loaded successfully.")

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (320, 240), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

time.sleep(2)

print("Capturing one frame...")
frame = picam2.capture_array()
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

print("Preparing blob...")
blob = cv2.dnn.blobFromImage(
    frame,
    scalefactor=1/255,
    size=(IMG_SIZE, IMG_SIZE),
    swapRB=True,
    crop=False
)

net.setInput(blob)

print("Running single inference... please wait")
start = time.time()
outputs = net.forward()
end = time.time()

print(f"Inference finished in {end - start:.2f} seconds")

if isinstance(outputs, tuple):
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
else:
    print("Output shape:", outputs.shape)

picam2.stop()
print("Done.")
