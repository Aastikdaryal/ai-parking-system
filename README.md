SMART PARKING DETECTION SYSTEM

An AI-powered Smart Parking Detection System that automatically detects free and occupied parking slots from images, recorded videos, and live webcam feed using Computer Vision and YOLOv8.
The system is optimized for real-time performance with CUDA-enabled GPU acceleration and is presented through an interactive Streamlit dashboard.

KEY FEATURES

Image-based parking detection
Recorded video-based parking detection
Real-time webcam feed support
Green bounding box indicates free parking slot
Red bounding box indicates occupied parking slot
GPU-accelerated YOLOv8 with CUDA support
Optimized for smooth real-time performance
Interactive Streamlit dashboard with multiple input modes

TECH STACK

Programming Language: Python
Computer Vision: OpenCV
Object Detection: YOLOv8 (Ultralytics)
Deep Learning Framework: PyTorch with CUDA
Dashboard and UI: Streamlit

SYSTEM ARCHITECTURE

User Interface (Streamlit Dashboard)
Image Upload | Video Upload | Live Webcam
↓
Input Processing
Image decoding, video frame extraction, webcam frame capture
↓
Automatic Slot Detection
Edge detection, contour analysis, parking slot localization
↓
Vehicle Detection Engine
YOLOv8 with GPU acceleration detects cars, bikes, buses
↓
Slot Occupancy Logic
Bounding box overlap check to classify free vs occupied slots
↓
Visual Output
Green boxes for free slots, red boxes for occupied slots with slot count display

INSTALLATION AND SETUP

Step 1: Clone the repository

git clone https://github.com/Aastikdaryal/ai-parking-system.git

cd ai-parking-system

Step 2: Install dependencies

pip install -r requirements.txt

Step 3: Install CUDA-enabled PyTorch (recommended for NVIDIA GPU)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Verify GPU availability

import torch
print(torch.cuda.is_available())

HOW TO RUN THE PROJECT

Launch the Streamlit dashboard

streamlit run dashboard.py

The dashboard provides three working modes

Image Upload Mode
Upload a parking image and detect free and occupied slots automatically

Video Upload Mode
Upload a recorded parking video and detect slot occupancy frame by frame

Live Webcam Mode
Real-time parking detection using a webcam feed

PERFORMANCE

CPU performance: approximately 15 to 20 frames per second
GPU performance with CUDA: approximately 30 to 45 frames per second

Performance optimizations include
Frame resizing
YOLO inference skipping
CUDA GPU acceleration
Reduced OpenCV buffering

USE CASES

Smart city parking management
Shopping malls and office complexes
University and campus parking systems
Traffic and urban infrastructure monitoring

PROJECT STRUCTURE

ai-parking-system
detector
parking_detector.py
auto_slot_detector.py
dashboard.py
phase6_video_ultra_smooth_gpu.py
requirements.txt
.gitignore
README.md

AUTHOR

Aastik Daryal
MCA
AI and Computer Vision Enthusiast
GitHub: https://github.com/Aastikdaryal
