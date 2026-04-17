# 🚀 Real-Time Human Detection & Tracking on Raspberry Pi

![Python](https://img.shields.io/badge/Python-3.9-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-red)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-orange)

---

## 📌 Project Overview

This project implements a **real-time human detection and tracking system** deployed on a **Raspberry Pi**.
It integrates **YOLO-based object detection**, lightweight tracking logic, and performance evaluation to operate under **edge computing constraints**.

The system detects humans, tracks their position within the frame, and generates movement decisions such as **LEFT, RIGHT, FORWARD, BACKWARD, and HOLD**.

---

## 🎯 Key Features

✔ Real-time human detection using YOLO (ONNX)
✔ Lightweight target tracking (centre-based)
✔ Behaviour decision module
✔ Edge deployment on Raspberry Pi
✔ Performance evaluation (FPS, resolution impact)
✔ Detection vs segmentation comparison
✔ Experimental logging and video recording

---

## 🧠 System Workflow

```
Camera Input → Object Detection → Target Tracking → Behaviour Decision → Output
```

---

## 🛠️ Tech Stack

| Component       | Technology Used    |
| --------------- | ------------------ |
| Programming     | Python             |
| Computer Vision | OpenCV             |
| Model           | YOLO (ONNX)        |
| Hardware        | Raspberry Pi 4     |
| Camera          | Pi Camera Module 3 |

---

## 📂 Project Structure

```
.
├── Results/
│   ├── frames/                # Extracted frames from experiments
│   ├── *.mp4 / *.avi         # Recorded experiment videos
│   ├── *.png                 # Sample outputs and screenshots
│
├── logs/
│   ├── *.csv                 # FPS logs and experiment data
│
├── models/
│   ├── yolov8n.onnx
│   ├── yolov8n_416.onnx
│   ├── yolov8n_seg_416.onnx
│
├── src/
│   ├── live_demo_tracking.py
│   ├── live_demo_target_lock.py
│   ├── live_demo_tracking_smooth.py
│   ├── seg_demo.py
│   ├── browser_stream_tracking.py
│   ├── record_tracking_experiment.py
│   ├── record_target_lock_experiment.py
│   └── other utility scripts
│
├── camera_pi_fps.py
├── camera_pi_fps_picamera2.py
├── yolo8_onnx_pi.py
├── yolo_pi_logger.py
├── requirements.txt
├── .gitignore
└── README.md
```


## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/live_demo_tracking.py
```

---

## 📊 Results Summary

* Average FPS: ~2.8–3.2 (depending on resolution)
* Performance is limited by CPU-based inference on Raspberry Pi
* Segmentation reduces FPS due to higher computational cost

---

## 📎 Dissertation Context

This project was developed as part of an MSc dissertation focused on **real-time computer vision systems on resource-constrained edge devices**.

---

## 📸 Sample Outputs



---

## 📜 License

This project is for academic and educational purposes.
