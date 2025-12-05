🧑‍🤝‍🧑 Face Recognition Pipeline (MediaPipe + LBPH)

This project implements a complete face-recognition system using MediaPipe for face detection and LBPH (Local Binary Patterns Histogram) for classical face recognition.
It demonstrates how a detection stage and a recognition stage work together in an AI-Without-Machine-Learning system.

🚀 Features

✔ Real-time face detection using MediaPipe

✔ LBPH face recognition (no deep learning)

✔ Supports multiple faces at the same time

✔ Works with at least two different people

✔ Simple scripts for dataset creation, training, and testing

📦 Requirements

Install dependencies:

pip install opencv-python opencv-contrib-python mediapipe numpy

📂 Project Structure
├── collect_dataset.py        # Create dataset using webcam
├── train_lbph.py             # Train LBPH model from dataset
├── recognize_live_multi.py   # Real-time multi-face recognition
├── dataset/                  # Auto-created dataset folders
├── lbph_model.yml            # Saved LBPH model (after training)
└── labels.json               # Label mapping

📝 How It Works
1. Face Detection (MediaPipe)

MediaPipe finds all faces in the camera frame and returns bounding boxes.

2. Face Preprocessing

Each detected face is:

Cropped

Converted to grayscale

Resized to 200×200 pixels

3. Recognition (LBPH)

LBPH compares the processed face to histogram patterns from the dataset and outputs:

The predicted person’s name

A confidence score (lower = better)

4. Multi-Face Output

Every face in the frame is labeled individually.

📸 Step 1 — Collect Dataset

Run:

python collect_dataset.py


Enter a name (example: john), look at the camera, and the script will save 100+ images to:

dataset/john/


Repeat for another person—at least two people are required.

🏋️ Step 2 — Train the LBPH Model

Run:

python train_lbph.py


This generates:

lbph_model.yml
labels.json

🎥 Step 3 — Run Multi-Face Recognition
python recognize_live_multi.py


The webcam window will open and show live predictions for all faces detected.

This proves the system recognizes two or more people correctly.

🎯 Assignment Requirements Check

✔ Detects faces → MediaPipe

✔ Recognizes at least 2 different people → LBPH

✔ Real-time multi-face handling

✔ Complete detection + recognition pipeline

✔ No machine learning model training (classical CV only)