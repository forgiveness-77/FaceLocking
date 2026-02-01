# Face Recognition with ArcFace ONNX and 5-Point Alignment

## Overview

This project provides a **modular, CPU-only, fully understandable face recognition system** built step-by-step using the **ArcFace ONNX model** with **5-point facial landmark alignment**.

The system is deliberately decomposed into independent, testable modules so that every stage (detection, landmark extraction, alignment, embedding, enrollment, recognition) can be validated separately.

**Designed for:**
- Educational purposes and learning
- Real-world practical use on ordinary laptops/desktops
- Transparency and debuggability (no black-box end-to-end frameworks)

This project upgrades from the author's previous work *Face Recognition without Deep Learning* (LBPH), now incorporating modern deep learning embeddings while maintaining CPU-friendliness and reproducibility.

### Key Features

- **Open-set recognition** using cosine similarity on ArcFace embeddings
- **5-point landmark-based alignment** (eyes, nose, mouth corners) for pose/scale correction
- **ONNX Runtime** for efficient CPU inference
- **Modular & testable** architecture
- **Automated project setup** via `init_project.py`
- **Cross-platform support**: macOS, Linux, Windows (no GPU required)

## Project Structure

```
face-recognition-5pt/
├── data/
│   ├── db/
│   │   ├── face_db.json        # metadata (names, timestamps, etc.)
│   │   └── face_db.npz         # L2-normalized embeddings
│   └── enroll/
│       ├── <Identity_Name>/
│       │   └── *.jpg           # aligned 112×112 enrollment images
├── models/
│   └── embedder_arcface.onnx   # ArcFace ONNX model
├── src/
│   ├── align.py                # Face alignment module
│   ├── camera.py               # Webcam feed handler
│   ├── detect.py               # Face detection
│   ├── embed.py                # ArcFace embedding extraction
│   ├── enroll.py               # Enrollment pipeline
│   ├── evaluate.py             # Threshold evaluation
│   ├── haar_5pt.py             # Haar cascade + 5-point detector
│   ├── landmarks.py            # Facial landmark detection
│   └── recognize.py            # Real-time recognition
├── init_project.py             # Project structure initialization
├── README.md
└── book/                       # (optional) book-related files
```

## Pipelines

### 1. Enrollment Pipeline
1. Face Detection  
2. 5-Point Landmark Detection  
3. Face Alignment (Warping to 112×112)  
4. ArcFace Embedding Extraction  
5. Store L2-normalized embedding in database

### 2. Recognition Pipeline
1. Face Detection  
2. 5-Point Landmark Detection  
3. Face Alignment  
4. ArcFace Embedding Extraction  
5. Compare with stored embeddings → threshold decision

## Setup Instructions

### Requirements
- Python 3.9+
- Webcam (for recognition modules)
- Supported OS: macOS, Linux, Windows

### Step 1: Create & Activate Virtual Environment

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Step 2: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install opencv-python numpy onnxruntime scipy tqdm mediapipe
```

### Step 3: Initialize Project Structure

```bash
python init_project.py
```

This script creates all necessary directories and placeholder files. Safe to re-run—never overwrites existing files.

### Step 4: Grant Camera Permissions

**macOS:** 
System Settings → Privacy & Security → Camera → Allow Terminal / VS Code

**Windows / Linux:** 
Ensure no other application is using your webcam

## Quick Validation

### Camera Check
```bash
python -m src.camera
```

Expected output:
- Live video window opens
- FPS counter displayed
- Smooth motion
- Press `q` to exit

If this fails, verify camera permissions and availability before proceeding.

## Module Testing Commands

Test individual components to validate setup:

```bash
# Camera feed and FPS benchmark
python -m src.camera

# Face detection with bounding boxes
python -m src.detect

# 5-point landmarks visualization
python -m src.landmarks

# Face alignment to 112×112
python -m src.align

# ArcFace embedding extraction
python -m src.embed

# Enroll identities into database
python -m src.enroll

# Evaluate and tune similarity threshold
python -m src.evaluate

# Live real-time recognition with webcam
python -m src.recognize
```

## Usage Workflow

1. **Enroll identities:** `python -m src.enroll`
   - Follow prompts to capture and register new faces
   - Embeddings saved to database

2. **Recognize faces:** `python -m src.recognize`
   - Real-time webcam feed with live recognition
   - Shows identity matches with confidence scores

3. **Evaluate threshold:** `python -m src.evaluate`
   - Fine-tune similarity threshold for your use case

## Troubleshooting

### Camera not detected
- Check permissions 
- Verify no other application is using the camera
- Try changing camera index in `src/camera.py`

### Poor recognition accuracy
- Ensure good lighting during enrollment
- Enroll multiple angles/poses per identity
- Adjust threshold in `src/evaluate.py`
- Verify faces are frontal (5-point alignment works best with minimal head tilt)

### Performance issues
- Use ONNX Runtime instead of other inference engines
- Reduce frame resolution for faster processing
- Limit number of identities in database


## License

Educational and non-commercial use encouraged.

---

Happy coding & face recognition experimenting! 🚀