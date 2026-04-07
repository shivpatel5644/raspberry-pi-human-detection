# 🚀 Real-Time Human Detection & Tracking on Raspberry Pi

![Python](https://img.shields.io/badge/Python-3.9-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-red)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-orange)

---

## 📌 Project Overview
This project implements a **real-time human detection and tracking system** deployed on a **Raspberry Pi**.  
It integrates **YOLO-based object detection**, tracking logic, and performance evaluation to operate under **edge computing constraints**.

The system detects humans, tracks their position within the frame, and generates movement decisions such as **left, right, forward, backward, or hold**.

---

## 🎯 Key Features
✔ Real-time human detection using YOLO (ONNX)  
✔ Target tracking and movement analysis  
✔ Edge deployment on Raspberry Pi  
✔ Performance evaluation (FPS, latency, resolution impact)  
✔ Detection vs segmentation comparison  
✔ Experimental logging and recording system  

---

## 🧠 System Workflow

---

## 🛠️ Tech Stack

| Component        | Technology Used |
|----------------|----------------|
| Programming     | Python         |
| Computer Vision | OpenCV         |
| Model           | YOLO (ONNX)    |
| Hardware        | Raspberry Pi 4 |
| Camera          | Pi Camera Module 3 |

---

## 📂 Project Structure

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/live_demo_tracking.py
