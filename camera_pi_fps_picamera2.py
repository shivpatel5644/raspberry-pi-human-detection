import time
from picamera2 import Picamera2

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

time.sleep(1)  # warmup

t0 = time.time()
frames = 0

while time.time() - t0 < 10:
    frame = picam2.capture_array()  # numpy array
    frames += 1

picam2.stop()

print("Frames in 10s:", frames)
print("Approx FPS:", frames / 10)
