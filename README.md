# 🚗 Smart Parking Detection System

An AI-powered smart parking system that detects **free and occupied parking slots**
from **images, videos, and live webcam feed** using **Computer Vision and YOLO**.

---

## 🔥 Features
- 📷 Image-based parking detection
- 🎥 Video-based parking detection
- 📡 Real-time webcam feed
- 🟩 Green box → Free slot
- 🟥 Red box → Occupied slot
- ⚡ GPU-accelerated YOLO (CUDA supported)
- 🖥️ Interactive Streamlit dashboard

---

## 🧠 Tech Stack
- Python
- OpenCV
- YOLOv8 (Ultralytics)
- PyTorch (CUDA)
- Streamlit

---

## 🏗️ System Architecture
┌───────────────────────────────┐
│         User Interface        │
│       (Streamlit Dashboard)   │
│                               │
│  • Image Upload               │
│  • Video Upload               │
│  • Live Webcam Feed           │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│        Input Processing        │
│                               │
│  • Image Decoder               │
│  • Video Frame Extractor       │
│  • Webcam Frame Capture        │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│     Auto Slot Detection        │
│   (Computer Vision Module)     │
│                               │
│  • Edge Detection              │
│  • Contour Analysis            │
│  • Parking Slot Localization  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   Vehicle Detection Engine     │
│        (YOLOv8 - GPU)          │
│                               │
│  • Car / Bike / Bus Detection │
│  • CUDA Accelerated Inference │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│     Slot Occupancy Logic       │
│                               │
│  • Bounding Box Overlap Check  │
│  • Free vs Occupied Decision  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│        Visual Output           │
│                               │
│  • Green Box → Free Slot       │
│  • Red Box → Occupied Slot    │
│  • Slot Count Display         │
└───────────────────────────────┘
