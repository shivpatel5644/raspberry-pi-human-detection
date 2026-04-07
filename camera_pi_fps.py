import cv2
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Camera not opened.")

t0 = time.time()
frames = 0

while time.time() - t0 < 10:
    ok, frame = cap.read()
    if not ok:
        break
    frames += 1

cap.release()
print("Frames in 10s:", frames)
print("Approx FPS:", frames / 10)
