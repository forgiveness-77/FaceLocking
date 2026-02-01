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
# Face Recognition System with Face Locking Feature

## Overview

This project provides a modular, CPU-only, fully understandable face recognition system with an advanced Face Locking feature. Built step-by-step using the ArcFace ONNX model with 5-point facial landmark alignment, the system not only recognizes faces but also tracks and monitors specific individuals over time, detecting their actions and expressions.

Designed for:

- Educational purposes and learning advanced computer vision concepts
- Real-world practical use on ordinary laptops/desktops (CPU-only)
- Transparency and debuggability (no black-box end-to-end frameworks)
- Real-time face tracking and behavior monitoring

This project upgrades from traditional face recognition by adding persistent identity tracking and action detection, moving from static recognition to dynamic behavior analysis.

### Key Features

- Open-set recognition using cosine similarity on ArcFace embeddings
- 5-point landmark-based alignment (eyes, nose, mouth corners) for pose/scale correction
- ONNX Runtime for efficient CPU inference
- Modular & testable architecture
- Automated project setup via `init_project.py`
- Cross-platform support: macOS, Linux, Windows (no GPU required)
- NEW: Face Locking - Persistent tracking of specific individuals
- NEW: Action Detection - Recognize movements and expressions in real-time
- NEW: Timeline Recording - Log all detected actions to timestamped files

## Project Structure

```
FaceRecognition/
├── data/
│   ├── db/
│   │   ├── face_db.json        # metadata (names, timestamps, etc.)
│   │   └── face_db.npz         # L2-normalized embeddings
│   ├── enroll/
│   │   ├── <Identity_Name>/
│   │   │   └── *.jpg           # aligned 112×112 enrollment images
│   ├── debug_aligned/          # Debug output from alignment module
│   ├── embeddings/             # Saved embedding vectors
│   └── lock_history/           # Action history files from face locking
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
│   ├── recognize.py            # Real-time recognition
│   ├── face_lock.py            # NEW: Face locking and tracking module
│   ├── action_detector.py      # NEW: Action and expression detection
│   └── lock_controller.py      # NEW: Main controller for face locking demo
├── init_project.py             # Project structure initialization
├── run_face_lock.py            # NEW: Launch face locking demo
├── README.md
└── requirements.txt
```

## NEW: Face Locking Feature

### What is Face Locking?

Face Locking is an advanced feature that:

- Recognizes a specific enrolled identity
- Locks onto that identity when they appear in the camera
- Tracks them persistently across frames
- Detects their actions and expressions
- Records a timeline of their behavior

Unlike basic face recognition which processes each frame independently, Face Locking maintains state and focuses on tracking a specific individual over time.

### Implemented Features

1. Manual Face Selection
   - Choose which enrolled identity to lock onto (e.g., "Gabi", "Fani", or "User")
   - System only tracks the selected identity
   - Other faces are ignored once lock is established

2. Face Locking Mechanism
   - When target face appears and is confidently recognized:
     - System displays "LOCKED: [Identity]" with visual indicator
     - Face bounding box changes color (green → purple)
     - Lock status is maintained across frames
     - Does not jump to other faces, even if they appear

3. Stable Tracking
   - Continues tracking the same face as it moves
   - Tolerates brief recognition failures (up to 5 seconds)
   - Releases lock only if face disappears for extended period
   - Uses spatial and temporal consistency checks

4. Action Detection (While Locked)
   - Detects and visualizes the following actions:

     Action	Detection Method	Visual Indicator
     Face Moved Left	Horizontal position tracking	← Arrow + Distance
     Face Moved Right	Horizontal position tracking	→ Arrow + Distance
     Eye Blink	Eye landmark distance changes	👁️ "Blink!" text
     Smile/Laugh	Mouth width/height ratio	😊 "Smile!" text
     Looking Up/Down	Vertical face position	↑↓ Arrows
     Nodding	Vertical movement pattern	"Nodding" text

5. Action History Recording
   - While face is locked, all detected actions are recorded to timestamped files:

     - File naming format: <face>_history_<timestamp>.txt
     - Example: gabi_history_20260129112099.txt
     - Format per entry: [timestamp] [action_type]: description
     - Location: data/lock_history/

   - Example history file:

```
[2025-01-30 14:30:15] LOCK_ACQUIRED: Locked onto identity 'Gabi'
[2025-01-30 14:30:17] MOVEMENT: Moved right by 120 pixels
[2025-01-30 14:30:19] EXPRESSION: Blink detected
[2025-01-30 14:30:22] EXPRESSION: Smile detected
[2025-01-30 14:30:25] MOVEMENT: Moved left by 80 pixels
[2025-01-30 14:30:30] LOCK_RELEASED: Face lost for 5+ seconds
```

## Pipelines

### 1. Enrollment Pipeline (Existing)
1. Face Detection

2. 5-Point Landmark Detection

3. Face Alignment (Warping to 112×112)

4. ArcFace Embedding Extraction

5. Store L2-normalized embedding in database

### 2. Recognition Pipeline (Existing)
1. Face Detection

2. 5-Point Landmark Detection

3. Face Alignment

4. ArcFace Embedding Extraction

5. Compare with stored embeddings → threshold decision

### 3. Face Locking Pipeline (NEW)
1. Face Detection & Recognition

2. Identity Verification (Is this the target identity?)

3. Lock Acquisition (If match confidence > threshold)

4. Continuous Tracking (Maintain lock across frames)

5. Action Detection (Monitor movements & expressions)

6. History Logging (Record all detected actions)

7. Lock Maintenance (Handle brief occlusions/losses)

8. Lock Release (Only after extended absence)

## Setup Instructions

### Requirements
- Python 3.9+
- Webcam (for recognition and locking modules)
- Supported OS: macOS, Linux, Windows

### Step 1: Create & Activate Virtual Environment

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows (Command Prompt):

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Step 2: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install opencv-python numpy onnxruntime scipy tqdm mediapipe
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
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
# Basic recognition modules
python -m src.camera              # Camera feed and FPS benchmark
python -m src.detect              # Face detection with bounding boxes
python -m src.landmarks           # 5-point landmarks visualization
python -m src.align               # Face alignment to 112×112
python -m src.embed               # ArcFace embedding extraction
python -m src.enroll              # Enroll identities into database
python -m src.evaluate            # Evaluate and tune similarity threshold
python -m src.recognize           # Live real-time recognition with webcam

# NEW: Face Locking modules
python src/face_lock.py           # Test face locking logic
python src/action_detector.py     # Test action detection
python run_face_lock.py           # Complete face locking demo
```

## Usage Workflow

1. Enroll Identities (Prerequisite)

```bash
python -m src.enroll
```

Follow prompts to capture and register faces

At least enroll one identity (e.g., "Gabi") for testing

Embeddings saved to database

2. Test Basic Recognition

```bash
python -m src.recognize
```

Verify your enrolled identity is recognized

Note the confidence scores

Adjust threshold if needed in `src/evaluate.py`

3. Run Face Locking Demo

```bash
python run_face_lock.py
```

Interface Controls:

```
[MAIN CONTROLS]
  L       : Toggle lock mode ON/OFF
  Space   : Select identity for locking
  T       : Toggle action detection
  R       : Reset lock and clear history
  H       : Show/hide help panel
  Q       : Quit application

[ACTION SHORTCUTS]
  1-6     : Manually trigger actions (for testing)
  B       : Simulate blink
  S       : Simulate smile
  M       : Simulate movement
```

Visual Indicators:

- Green box: Detected face (unlocked)
- Purple box: Locked target face
- Red box: Unknown/other face
- Status bar: Lock status, target identity, action count
- Action log: Recent detected actions scroll on screen
- Movement arrows: Show direction and magnitude of movement
- Expression icons: Appear when blinks/smiles detected

4. Review Action History

All actions are saved to data/lock_history/

Files are automatically created with timestamp

Each file contains chronological record of all detected actions

Review with any text editor

## How Face Locking Works

### Technical Implementation

Identity Selection:

- User selects which enrolled identity to track
- System loads corresponding embedding from database
- Threshold set at 0.5 cosine similarity for recognition

Lock Acquisition:

- When target appears: confidence > threshold for 3 consecutive frames
- Bounding box color changes to purple
- "LOCKED" status displayed
- Position tracking initialized

Stable Tracking:

- Uses Kalman filter for smooth position prediction
- Maintains lock through brief recognition failures (5-second grace)
- Spatial consistency check prevents jumping to other faces
- Only releases after extended absence (>5 seconds)

Action Detection:

- Movement: Tracks horizontal face position changes
- Blink: Monitors vertical distance between eyelids
- Smile: Analyzes mouth width-to-height ratio
- Nodding: Detects vertical oscillation pattern

History Recording:

- Creates new file when lock is acquired
- Appends each detected action with timestamp
- Saves file when lock is released
- Files are human-readable for analysis

## Advanced Configuration

### Adjusting Detection Sensitivity

Edit `src/action_detector.py`:

```python
# Movement detection
MOVEMENT_THRESHOLD = 50  # pixels to register as movement

# Blink detection
BLINK_RATIO_THRESHOLD = 0.3  # eye aspect ratio threshold

# Smile detection
SMILE_RATIO_THRESHOLD = 1.5  # mouth width/height ratio
```

### Lock Behavior Tuning

Edit `src/face_lock.py`:

```python
# Lock acquisition
CONSECUTIVE_MATCHES = 3  # Frames needed to acquire lock
MATCH_THRESHOLD = 0.5    # Cosine similarity threshold

# Lock maintenance
MAX_LOCK_LOSS_FRAMES = 150  # ~5 seconds at 30 FPS
POSITION_TOLERANCE = 50     # Pixels for position matching
```

## Troubleshooting

### Camera not detected
- Check permissions
- Verify no other application is using the camera
- Try changing camera index in `run_face_lock.py`

### Poor recognition/locking accuracy
- Ensure good lighting during enrollment
- Enroll multiple angles/poses per identity
- Adjust threshold in `src/face_lock.py`
- Verify faces are frontal (5-point alignment works best with minimal head tilt)

### Performance issues
- Reduce frame resolution in `run_face_lock.py`
- Limit number of enrolled identities
- Disable action detection if not needed

### Action detection too sensitive/insensitive
- Adjust thresholds in `src/action_detector.py`
- Calibrate with your specific movements

## Files Created by Face Locking

### Action History Files (data/lock_history/):

- Named: <identity>_history_YYYYMMDD_HHMMSS.txt
- Contains chronological action log
- Human-readable format

### Debug Images (data/debug_aligned/):

- Aligned face images saved when pressing 'S'
- Useful for debugging alignment issues

### Embedding Files (data/embeddings/):

- Saved embedding vectors for analysis

## Educational Value

This project demonstrates:

- Stateful computer vision (vs. frame-by-frame processing)
- Multi-object tracking concepts
- Temporal filtering for robust tracking
- Action recognition from geometric features
- System design with modular components

## Future Extensions

Potential enhancements:

- Multiple face locking (track several identities simultaneously)
- Advanced action detection (head tilts, eyebrow raises)
- Emotion recognition integration
- Voice command for lock control
- Network streaming for remote monitoring
- Database integration for long-term behavior analysis

## License

Educational and non-commercial use encouraged.

---

## Quick Start Summary

Setup: pip install -r requirements.txt

Enroll: python -m src.enroll (register at least one face)

Test: python -m src.recognize (verify recognition works)

Lock: python run_face_lock.py (launch face locking demo)

Select: Press Space to choose identity, L to toggle lock

Monitor: Watch actions get detected and recorded

Review: Check data/lock_history/ for saved timelines

Happy coding & face tracking! 🚀