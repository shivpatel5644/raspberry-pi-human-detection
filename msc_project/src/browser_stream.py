from flask import Flask, Response
from threading import Thread, Lock
from picamera2 import Picamera2
import cv2
import time

# =========================
# Flask streaming setup
# =========================
app = Flask(__name__)
output_frame = None
frame_lock = Lock()

def generate():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            frame = output_frame

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask():
    app.run(host='0.0.0.0', port=5000, threaded=True)

# =========================
# Your main vision pipeline
# =========================
def main():
    global output_frame

    flask_thread = Thread(target=start_flask, daemon=True)
    flask_thread.start()

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(
            main={"size": (640, 480)}
        )
    )
    picam2.start()

    while True:
        frame = picam2.capture_array()

        # --------------------------------------------------
        # PUT YOUR EXISTING DETECTION / TRACKING CODE HERE
        # Example:
        # frame = run_detection(frame)
        # frame = run_tracking(frame)
        # frame = draw_behaviour(frame)
        # --------------------------------------------------

        # Example overlay just to test
        cv2.putText(frame, "Live Tracking Stream", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Optional local preview on Pi
        cv2.imshow("Tracking", frame)

        # Encode processed frame for browser stream
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            with frame_lock:
                output_frame = jpeg.tobytes()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
