# AI Vision Studio 👁️⚡

A modern, interactive Computer Vision & AI web studio built with **Python**, **Flask**, **OpenCV**, and **JavaScript**.

![AI Vision Studio](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0-red.svg)
![Deep Learning](https://img.shields.io/badge/YOLO-v5%20Nano-green.svg)
![SQLite](https://img.shields.io/badge/Database-SQLite3-lightgrey.svg)
![Theme](https://img.shields.io/badge/Theme-White%20%26%20Dark%20Blue-0b152d.svg)

---

## 📖 1. Project Overview

**AI Vision Studio** allows students, engineers, and developers to explore fundamental computer vision techniques and deep learning inference directly inside a modern web browser. 

Users can upload images or use their live webcam feed to:
- Transform images using OpenCV filters (blur, edges, thresholding, brightness, contrast, and artistic effects).
- Detect faces and eyes using classical **Viola-Jones Haar Cascades**.
- Run real-time **YOLO Deep Neural Network Object Detection** across 80 common object categories.
- Stream their webcam feed with real-time browser canvas filters and capture snapshots for server-side OpenCV analysis.
- Inspect persistent analysis logs and benchmark metrics saved inside **SQLite**.

---

## 🗂️ 2. Project Folder Structure

```text
ai-vision-studio/
├── app.py                      # Flask web server, API routes & upload security
├── vision.py                   # Computer vision engine (OpenCV, Haar, YOLOv5 DNN)
├── database.py                 # SQLite database helper for logging & stats
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation & setup instructions
├── history.db                  # Local SQLite database (auto-generated)
├── templates/                  # Jinja2 HTML5 UI templates
│   ├── base.html               # Shared responsive layout, navbar & footer
│   ├── index.html              # Studio Dashboard with live stats & cards
│   ├── image_processing.html   # Interactive filter playground with split slider
│   ├── face_detection.html     # Haar cascade face & eye detector + avatar cards
│   ├── webcam.html             # Live browser webcam studio & canvas filters
│   ├── object_detection.html   # Pretrained YOLOv5 AI object detection suite
│   └── history.html            # Complete persistent analysis history table
├── static/
│   ├── css/
│   │   └── style.css           # Premium AI Studio design system (white & dark-blue)
│   ├── js/
│   │   └── script.js           # Split-slider comparison, canvas loop & AJAX
│   ├── uploads/                # User uploaded images and webcam snapshots
│   ├── results/                # Annotated and filtered output images
│   └── samples/                # Included test images (portraits, sports, scenes)
└── models/                     # Pretrained neural networks & cascades
    ├── haarcascade_frontalface_default.xml   # Haar cascade for face detection
    ├── haarcascade_eye.xml                   # Haar cascade for eye detection
    └── yolov5n.onnx                          # YOLOv5 Nano ONNX model (3.98 MB)
```

---

## 🔍 3. File-by-File Technical Explanation

### 1. `app.py`
The central controller of the web application.
- Configures upload directories, secure unique filename generation (`uuid`), and file size limits (16 MB max).
- Validates allowed extensions (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`).
- Exposes clean REST API endpoints (`/api/process-image`, `/api/detect-faces`, `/api/detect-objects`, `/api/webcam-capture`, `/api/history/...`).
- Converts base64 webcam snapshots from browser canvas to OpenCV image files.
- Serves processed images securely with file download headers (`/download/<folder>/<filename>`).

### 2. `vision.py`
The computer vision engine containing optimized OpenCV pipelines:
- **`apply_image_playground_filters()`**: Performs brightness/contrast scaling (`convertScaleAbs`), Gaussian/Box/Median blurs, Canny/Sobel/Laplacian edge operators, Otsu and adaptive binarization, rotations, flips, sharpening, sepia, invert, and pencil sketch effects.
- **`detect_faces()`**: Uses `cv2.CascadeClassifier` with multi-scale scanning (`scaleFactor`, `minNeighbors`). Draws futuristic HUD corner bounding boxes, isolates face regions of interest (ROI) to detect eyes, and crops face thumbnails.
- **`detect_objects_dnn()`**: Loads `yolov5n.onnx` into `cv2.dnn.readNetFromONNX()`. Preprocesses the image into a 640x640 blob, decodes predictions, applies Non-Maximum Suppression (`cv2.dnn.NMSBoxes`), and annotates class tags and confidence percentages.

### 3. `database.py`
The data persistence layer utilizing Python's built-in `sqlite3` module:
- Creates the `analysis_history` table automatically on boot.
- Records timestamp, module name, original/processed filenames, parameters, detection counts, file size in KB, and processing time in milliseconds.
- Computes aggregate metrics for dashboard stat cards (total runs, faces detected, objects detected).

### 4. `templates/`
Semantic HTML5 templates styled with the **AI Studio** theme:
- **Interactive Split Slider**: A draggable divider lets users visually wipe across the image to inspect the before-and-after difference.
- **Side-by-Side Dual View**: Displays original and processed images alongside resolution and size tags.
- **Avatar Cards**: Displays detected faces as cropped avatar badges with dimensions and eye counts.

### 5. `static/js/script.js`
Frontend interactivity and multimedia handling:
- In-browser webcam stream acquisition via `navigator.mediaDevices.getUserMedia()`.
- Real-time client-side filter simulation on HTML5 `<canvas>` via `requestAnimationFrame` loop.
- Instant snapshot capture converting canvas frames to JPEG blobs for server-side OpenCV processing.

---

## 💻 4. Installation & Setup Guide (Windows)

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14.
- A modern web browser (Google Chrome, Microsoft Edge, or Mozilla Firefox).

### Step 1: Open PowerShell or Command Prompt
Navigate to the directory:
```powershell
cd "c:\Users\Sai Susanth.K\PycharmProjects\ai-vision-studio"
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Packages
```powershell
python -m pip install -r requirements.txt
```

> **Note on Windows pip execution:** If `pip.exe` is blocked by an Application Control policy, always use `python -m pip install ...` which bypasses script execution restrictions.

---

## 🤖 5. Pretrained Models Configuration

The project uses lightweight, high-performance models stored directly in the `models/` directory:

| Model File | Source | Description | Size |
| :--- | :--- | :--- | :--- |
| `haarcascade_frontalface_default.xml` | OpenCV Official Repo | Classical Haar cascade for frontal human faces | 930 KB |
| `haarcascade_eye.xml` | OpenCV Official Repo | Classical Haar cascade for eye detection within face ROI | 341 KB |
| `yolov5n.onnx` | Ultralytics YOLOv5 | YOLOv5 Nano neural network trained on COCO (80 classes) | 3.98 MB |

### How Models Are Loaded
- **Haar Cascades**: Loaded via `cv2.CascadeClassifier(path)`.
- **YOLOv5 ONNX**: Loaded natively into memory via `cv2.dnn.readNetFromONNX(path)`. It requires **zero PyTorch or GPU installation**, running at ~250–350ms on standard CPUs.

---

## 🚀 6. Running the Application

Start the Flask development server:
```powershell
python app.py
```

You will see:
```text
==================================================
🚀 AI Vision Studio is launching on http://127.0.0.1:5000
==================================================
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🧪 7. Testing Each Feature Step-by-Step

### A. Testing Image Processing Playground (`/image-processing`)
1. Click **Playground** in the navigation bar.
2. Click one of the bundled sample pills at the bottom of the left sidebar (e.g., `nature_butterfly.jpg`).
3. Adjust the **Brightness** (+30) or **Contrast** (1.5x) sliders.
4. Select **Edge Detection -> Canny** or **Color Conversion -> Warm Sepia Vintage Tone**.
5. Click **Apply Filters**.
6. Drag the vertical divider on the image left and right to inspect the split comparison.
7. Click **Download Result** to save the processed file.

### B. Testing Face Detection (`/face-detection`)
1. Click **Face Detection** in the navigation bar.
2. Select the bundled `portrait_face.jpg` sample (or upload your own group photo).
3. Check the **Enable Eye Localization** checkbox.
4. Click **Detect Faces**.
5. Observe the bounding box and the generated **Face #1 Avatar** showing resolution and eye detections.
6. Review the educational callout on the difference between detecting face patterns versus recognizing identity.

### C. Testing Object Detection (`/object-detection`)
1. Click **Object Detection** in the navigation bar.
2. Click the `street_sports.jpg` sample image.
3. Keep the **Confidence Threshold** at 35%.
4. Click **Run AI Object Detection**.
5. In under 300ms, the system will identify the soccer players (`person`) and the soccer ball (`sports ball`) with bounding boxes and confidence percentages.

### D. Testing Webcam Studio (`/webcam`)
1. Click **Webcam Studio** in the navigation bar.
2. Click **Start Camera** and click **Allow** on the browser permission prompt.
3. Change the **Mode** dropdown to `Realtime Grayscale` or `Realtime Negative` to view instant canvas filtering.
4. Switch to `OpenCV Face Detection` and click **Capture Snapshot**.
5. The high-resolution camera frame is sent to the OpenCV backend, analyzed, and displayed below with a download button.

### E. Testing Analysis History (`/history`)
1. Click **History** in the navigation bar.
2. All your previous operations, run times (in ms), detected item counts, and image thumbnails are listed.
3. Test deleting individual logs or clicking **Clear All History**.

---

## 🔧 8. Troubleshooting & FAQ

### 1. Camera permission was denied in browser
- **Cause**: Browser blocked camera access.
- **Fix**: Click the tune/lock icon in your browser address bar (next to `http://127.0.0.1:5000`), ensure **Camera** is set to **Allow**, and refresh the page.

### 2. "Address already in use" (Port 5000 occupied)
- **Cause**: Another app (or previous Flask instance) is using port 5000.
- **Fix**: Open `app.py` and change the last line:
  ```python
  app.run(debug=True, host='127.0.0.1', port=5050)
  ```
  Then access `http://127.0.0.1:5050`.

### 3. Face detection detected 0 faces on my image
- **Fix**: Ensure the face is looking forward with good lighting. In the left panel, lower the **Scale Factor** to `1.05` and decrease **Min Neighbors** to `3` to increase detection sensitivity.

---

## 🎓 9. Educational Concepts: Computer Vision Fundamentals

1. **Spatial Filtering**: Convolution operations like Gaussian Blurring apply weighted matrix kernels to smooth noise by averaging neighboring pixels.
2. **Gradients & Edges**: Canny Edge Detection computes directional intensity gradients using derivative approximations (Sobel filters) followed by non-maximum suppression and hysteresis thresholding.
3. **Haar Wavelet Features**: Invented by Viola and Jones (2001), Haar cascades evaluate adjacent rectangular regions to detect facial contrast patterns in constant time using *Integral Images*.
4. **Single-Shot Object Detection**: YOLO divides an image into a grid and predicts bounding boxes with class confidence scores across the entire picture in one unified neural network pass.
