# LITA — Local Intelligent Threat Assessment (V1.0)

LITA (Local Intelligent Threat Assessment) is an advanced, high-performance edge computer vision application designed for real-time local hazard evaluation, weapon detection, and facial verification. 

By combining cutting-edge object detection and facial embeddings on a local multi-threaded pipeline, LITA provides microsecond latency assessment with a professional head-up display (HUD).

---

## Key Features

- **Local Object Detection (YOLOv8)**: Calibrated for critical target classes (person, knife, scissors) with precise per-class confidence thresholds to minimize false positives.
- **Biometric Identity Verification**: Real-time facial recognition powered by deep neural networks (`Facenet512` / `ArcFace`). Automatically handles portrait matching of whitelisted individuals.
- **Temporal Inference Caching**: Smart caching of biometric matches with high-precision time-to-live (TTL) limits. This optimization bypasses heavy facial recognition on consecutive frames, unlocking up to **30+ FPS edge throughput**.
- **Dynamic Threat HUD**: Visual alert bounding boxes:
  - **Green (KNOWN)**: Whitelisted, authenticated individuals.
  - **White (UNKNOWN)**: Unidentified individuals.
  - **Red (CRITICAL THREAT)**: Confirmed weapon classes.
- **Heuristic Threat Score Engine**: Calculates a continuous threat score (0-100) based on detected elements, immediately categorizing the environment status:
  - **SAFE** (0 - 29)
  - **ELEVATED** (30 - 59)
  - **CRITICAL** (60 - 100)

---

## Architecture & Technology Stack

LITA runs 100% locally on your machine, ensuring data privacy and offline readiness.

- **Core Engine**: Python 3.10+
- **Object Detection**: Ultralytics YOLOv8 (using dynamic CUDA/PyTorch half-precision execution)
- **Face Verification**: DeepFace (`Facenet512` or `ArcFace` backends)
- **Computer Vision & HUD**: OpenCV
- **Multi-threaded Ingestion**: Asynchronous frame acquisition to prevent UI blocking and thread starvation.

---

## Project Structure

```
lita/
├── whitelist/              # Directory for authorized face portraits (e.g. jane_doe.jpg)
│   └── README.md           # Guidelines for whitelist administration
├── .gitignore              # Ignores local venvs, weights, caches, and portraits
├── config.py               # Complete project configuration, model choices, and scoring weights
├── engine.py               # HUD renderer and threat calculation engine
├── inference.py            # YOLO and DeepFace inference controller with embedding cache
├── main.py                 # Application entrypoint (asynchronous frame processing & HUD loop)
├── stream.py               # Asynchronous video stream thread using OpenCV
├── verify_face.py          # Integration verification for facial verification pipeline
├── verify_threat.py        # Unit test suite for heuristic threat scoring scenarios
├── verify_yolo.py          # Validation script for YOLO class configurations
└── verify_stream.py        # Pipeline validation for camera stream acquisition
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- NVIDIA CUDA-capable GPU (highly recommended for real-time performance) with CUDA Toolkit installed.

### 2. Clone the Repository
```bash
git clone https://github.com/ZaviQ7/LITA.git
cd LITA
```

### 3. Initialize Virtual Environment & Install Dependencies
Create a clean virtual environment and install the required machine learning and computer vision libraries:
```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate

# Install requirements:
pip install opencv-python ultralytics deepface tf-keras torch torchvision torchaudio
```

### 4. Administer Whitelist Portraits
Drop `.jpg` or `.png` portraits of authenticated persons inside the `./whitelist` directory:
- Name the files with the person's name (e.g., `jane_doe.jpg`).
- LITA will auto-detect, compute embeddings, and display the name as the authenticated HUD label (e.g., `KNOWN: Jane Doe`) during runtime.

---

## Usage

Start the main LITA application:
```bash
python main.py
```

### Key Controls
- Press **`q`** inside the display window to exit LITA.

---

## Verification & Testing

LITA includes built-in verification tools to ensure all subsystems (camera, YOLO model, face verification, decision heuristics) are configured correctly.

### Run Decision Engine Tests
Verify that all mathematical scoring thresholds and status assignments match expected specs:
```bash
python verify_threat.py
```

### Run Stream Loop Tests
Validate OpenCV multi-threaded camera acquisition and framerate calculation:
```bash
python verify_stream.py
```

### Run YOLO Configuration Tests
Confirm that the default YOLO class mappings align with LITA configuration:
```bash
python verify_yolo.py
```

---

## License

This project is licensed under the MIT License.
