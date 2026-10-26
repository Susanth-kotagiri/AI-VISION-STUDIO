"""
AI Vision Studio - Computer Vision Processing Engine
Contains modular functions for image transformations, Haar Cascade face detection,
Pretrained Object Detection (MobileNet-SSD / OpenCV DNN), and MediaPipe analysis.
"""

import os
import time
import json
import cv2
import numpy as np

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'static', 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# Pretrained Model Paths
HAAR_FACE_PATH = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')
HAAR_EYE_PATH = os.path.join(MODELS_DIR, 'haarcascade_eye.xml')
YOLO_ONNX_PATH = os.path.join(MODELS_DIR, 'yolov5n.onnx')

# 80 COCO Classes for YOLO
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Color map for classes
CLASS_COLORS = {
    "person": (255, 99, 71),        # Tomato Red
    "car": (50, 205, 50),          # Lime Green
    "bicycle": (30, 144, 255),      # Dodger Blue
    "dog": (255, 165, 0),          # Orange
    "cat": (186, 85, 211),         # Orchid
    "chair": (218, 165, 32),       # Goldenrod
    "bottle": (0, 206, 209),       # Dark Turquoise
    "bus": (255, 20, 147),         # Deep Pink
    "sports ball": (0, 230, 255),  # Cyan
    "laptop": (138, 43, 226)       # Blue Violet
}


def allowed_file(filename):
    """Check if uploaded file has an allowed image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_image_safe(file_path):
    """
    Safely reads an image using numpy fromfile to support UTF-8 Windows paths.
    Returns BGR OpenCV image or None.
    """
    if not os.path.exists(file_path):
        return None
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading image {file_path}: {e}")
        return None


def save_image_safe(image, file_path):
    """
    Safely saves an image using imencode and tofile to support UTF-8 Windows paths.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if not ext:
            ext = '.jpg'
        success, encoded = cv2.imencode(ext, image)
        if success:
            encoded.tofile(file_path)
            return True
        return False
    except Exception as e:
        print(f"Error saving image {file_path}: {e}")
        return False


def get_image_metadata(file_path):
    """Returns dimensions, channels, and file size in KB."""
    if not os.path.exists(file_path):
        return None
    size_kb = round(os.path.getsize(file_path) / 1024, 2)
    img = read_image_safe(file_path)
    if img is None:
        return {'size_kb': size_kb, 'width': 0, 'height': 0, 'channels': 0}
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1
    return {
        'size_kb': size_kb,
        'width': w,
        'height': h,
        'channels': c,
        'aspect_ratio': round(w / h, 2) if h > 0 else 1.0
    }


# =====================================================================
# 1. Image Processing Playground
# =====================================================================
def apply_image_playground_filters(input_path, output_path, params):
    """
    Applies comprehensive OpenCV filters and transformations based on user parameters.
    Returns:
        tuple: (success, result_metadata, processing_time_ms)
    """
    start_time = time.time()
    img = read_image_safe(input_path)
    if img is None:
        return False, {"error": "Failed to load source image"}, 0.0

    original_h, original_w = img.shape[:2]
    processed = img.copy()

    # 1. Resizing
    resize_mode = params.get('resize_mode', 'none')
    if resize_mode == 'custom':
        tw = int(params.get('target_width', original_w))
        th = int(params.get('target_height', original_h))
        if 10 <= tw <= 4000 and 10 <= th <= 4000:
            processed = cv2.resize(processed, (tw, th), interpolation=cv2.INTER_AREA if tw < original_w else cv2.INTER_LANCZOS4)
    elif resize_mode == 'scale':
        scale_pct = float(params.get('scale_percent', 100)) / 100.0
        if 0.1 <= scale_pct <= 3.0 and scale_pct != 1.0:
            nw = max(10, int(original_w * scale_pct))
            nh = max(10, int(original_h * scale_pct))
            processed = cv2.resize(processed, (nw, nh), interpolation=cv2.INTER_AREA if scale_pct < 1.0 else cv2.INTER_LINEAR)

    # 2. Rotation & Flip
    rotation = int(params.get('rotation', 0))
    if rotation == 90:
        processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        processed = cv2.rotate(processed, cv2.ROTATE_180)
    elif rotation == 270:
        processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)

    flip_mode = params.get('flip', 'none')
    if flip_mode == 'horizontal':
        processed = cv2.flip(processed, 1)
    elif flip_mode == 'vertical':
        processed = cv2.flip(processed, 0)
    elif flip_mode == 'both':
        processed = cv2.flip(processed, -1)

    # 3. Brightness & Contrast
    # alpha: contrast factor [0.1, 3.0], beta: brightness [-100, 100]
    contrast = float(params.get('contrast', 1.0))
    brightness = float(params.get('brightness', 0.0))
    if contrast != 1.0 or brightness != 0.0:
        processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=brightness)

    # 4. Color conversions / Artistic effects
    color_mode = params.get('color_mode', 'original')
    if color_mode == 'grayscale':
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif color_mode == 'sepia':
        # Sepia transformation kernel
        kernel_sepia = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        processed = cv2.transform(processed, kernel_sepia)
        processed = np.clip(processed, 0, 255).astype(np.uint8)
    elif color_mode == 'invert':
        processed = cv2.bitwise_not(processed)
    elif color_mode == 'sharpen':
        sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        processed = cv2.filter2D(processed, -1, sharpen_kernel)
    elif color_mode == 'sketch':
        gray_sketch, color_sketch = cv2.pencilSketch(processed, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        processed = cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2BGR)

    # 5. Blur
    blur_type = params.get('blur_type', 'none')
    blur_ksize = int(params.get('blur_intensity', 0))
    if blur_ksize > 0:
        # Kernel size must be odd and >= 1
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        if blur_type == 'gaussian':
            processed = cv2.GaussianBlur(processed, (k, k), 0)
        elif blur_type == 'box':
            processed = cv2.blur(processed, (k, k))
        elif blur_type == 'median':
            processed = cv2.medianBlur(processed, k)

    # 6. Edge Detection
    edge_type = params.get('edge_type', 'none')
    if edge_type == 'canny':
        low_t = int(params.get('canny_low', 50))
        high_t = int(params.get('canny_high', 150))
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed
        edges = cv2.Canny(gray, low_t, high_t)
        processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif edge_type == 'sobel':
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(sobelx, sobely)
        mag = np.uint8(np.clip(mag / mag.max() * 255, 0, 255)) if mag.max() > 0 else np.uint8(mag)
        processed = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
    elif edge_type == 'laplacian':
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.uint8(np.absolute(lap))
        processed = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

    # 7. Thresholding (applied on grayscale)
    thresh_type = params.get('threshold_type', 'none')
    if thresh_type != 'none':
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed
        thresh_val = int(params.get('threshold_val', 127))
        if thresh_type == 'binary':
            _, thresh_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            processed = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
        elif thresh_type == 'binary_inv':
            _, thresh_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            processed = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
        elif thresh_type == 'otsu':
            _, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
        elif thresh_type == 'adaptive_mean':
            thresh_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            processed = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
        elif thresh_type == 'adaptive_gaussian':
            thresh_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            processed = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image_safe(processed, output_path)

    elapsed_ms = (time.time() - start_time) * 1000.0
    final_h, final_w = processed.shape[:2]

    meta = {
        'width': final_w,
        'height': final_h,
        'size_kb': round(os.path.getsize(output_path) / 1024, 2) if os.path.exists(output_path) else 0.0,
        'filters_applied': {k: v for k, v in params.items() if v not in ['none', 0, 1.0, '', 'original']}
    }
    return True, meta, elapsed_ms


# =====================================================================
# 2. Haar Cascade Face Detection
# =====================================================================
def detect_faces(input_path, output_path, scale_factor=1.1, min_neighbors=5, min_size=30, detect_eyes=False):
    """
    Performs face detection using OpenCV Haar Cascades.
    Draws modern bounding boxes and face counter tags.
    Optionally detects eyes within detected face regions.
    """
    start_time = time.time()
    img = read_image_safe(input_path)
    if img is None:
        return False, {"error": "Image could not be loaded"}, 0.0

    if not os.path.exists(HAAR_FACE_PATH):
        return False, {"error": f"Haar Cascade model missing at {HAAR_FACE_PATH}"}, 0.0

    face_cascade = cv2.CascadeClassifier(HAAR_FACE_PATH)
    if face_cascade.empty():
        return False, {"error": "Failed to initialize OpenCV Haar Face Cascade"}, 0.0

    eye_cascade = None
    if detect_eyes and os.path.exists(HAAR_EYE_PATH):
        eye_cascade = cv2.CascadeClassifier(HAAR_EYE_PATH)

    annotated = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Enhance contrast for better detection

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=float(scale_factor),
        minNeighbors=int(min_neighbors),
        minSize=(int(min_size), int(min_size)),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    detected_faces_list = []
    face_crops_info = []

    # Modern visual colors
    box_color = (255, 140, 0)      # Vivid Blue/Orange in BGR: (0, 140, 255) -> Deep sky orange/blue
    primary_color = (255, 120, 30)  # Neon Cyan/Azure
    tag_bg_color = (24, 28, 36)     # Dark slate
    text_color = (255, 255, 255)

    for idx, (x, y, w, h) in enumerate(faces, 1):
        # Draw sleek rectangle with corner accents
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 128, 0), 2)

        # Corner bracket accents for futuristic HUD look
        line_len = min(int(w * 0.2), 25)
        # Top-left
        cv2.line(annotated, (x, y), (x + line_len, y), (0, 220, 255), 4)
        cv2.line(annotated, (x, y), (x, y + line_len), (0, 220, 255), 4)
        # Top-right
        cv2.line(annotated, (x + w, y), (x + w - line_len, y), (0, 220, 255), 4)
        cv2.line(annotated, (x + w, y), (x + w, y + line_len), (0, 220, 255), 4)
        # Bottom-left
        cv2.line(annotated, (x, y + h), (x + line_len, y + h), (0, 220, 255), 4)
        cv2.line(annotated, (x, y + h), (x, y + h - line_len), (0, 220, 255), 4)
        # Bottom-right
        cv2.line(annotated, (x + w, y + h), (x + w - line_len, y + h), (0, 220, 255), 4)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - line_len), (0, 220, 255), 4)

        # Badge label
        label = f"Face #{idx}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tag_y = max(y - 8, label_h + 8)
        cv2.rectangle(annotated, (x, tag_y - label_h - 6), (x + label_w + 10, tag_y + 4), (20, 24, 34), -1)
        cv2.putText(annotated, label, (x + 5, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)

        # Optional eye detection within face ROI
        eyes_detected = 0
        if eye_cascade and not eye_cascade.empty():
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = annotated[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes:
                eyes_detected += 1
                center = (int(ex + ew / 2), int(ey + eh / 2))
                radius = int(round((ew + eh) * 0.25))
                cv2.circle(roi_color, center, radius, (0, 255, 127), 2)

        # Save cropped face avatar thumbnail
        crop_filename = f"crop_face_{idx}_{int(time.time()*1000)}.jpg"
        crop_path = os.path.join(RESULTS_DIR, crop_filename)
        # Pad crop slightly
        pad = int(min(w, h) * 0.1)
        cy1, cy2 = max(0, y - pad), min(img.shape[0], y + h + pad)
        cx1, cx2 = max(0, x - pad), min(img.shape[1], x + w + pad)
        face_crop = img[cy1:cy2, cx1:cx2]
        if face_crop.size > 0:
            save_image_safe(face_crop, crop_path)
            face_crops_info.append(crop_filename)

        detected_faces_list.append({
            'id': idx,
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h),
            'eyes_detected': eyes_detected,
            'crop_file': crop_filename if face_crop.size > 0 else None
        })

    # Save output annotated image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image_safe(annotated, output_path)

    elapsed_ms = (time.time() - start_time) * 1000.0

    result = {
        'face_count': len(faces),
        'faces': detected_faces_list,
        'scale_factor': scale_factor,
        'min_neighbors': min_neighbors,
        'eyes_enabled': detect_eyes,
        'note': "Haar Cascade detects contrast-based facial patterns, not identity."
    }
    return True, result, elapsed_ms


# =====================================================================
# 3. Pretrained Object Detection (OpenCV DNN & YOLOv5 ONNX)
# =====================================================================
_yolo_net = None


def get_yolo_network():
    """Singleton loader for YOLOv5 ONNX network."""
    global _yolo_net
    if _yolo_net is None:
        if os.path.exists(YOLO_ONNX_PATH) and os.path.getsize(YOLO_ONNX_PATH) > 1000000:
            try:
                _yolo_net = cv2.dnn.readNetFromONNX(YOLO_ONNX_PATH)
            except Exception as e:
                print(f"Failed to load YOLO net: {e}")
                _yolo_net = None
    return _yolo_net


def detect_objects_dnn(input_path, output_path, conf_threshold=0.35, nms_threshold=0.45):
    """
    Detects objects using OpenCV DNN with pretrained YOLOv5 ONNX.
    Supports 80 COCO classes with Non-Maximum Suppression (NMS).
    """
    start_time = time.time()
    img = read_image_safe(input_path)
    if img is None:
        return False, {"error": "Image could not be read"}, 0.0

    net = get_yolo_network()
    if net is None:
        return False, {"error": "YOLO model missing or could not be loaded from models/yolov5n.onnx"}, 0.0

    h, w = img.shape[:2]
    annotated = img.copy()

    # Create 640x640 blob for YOLOv5
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()[0]

    boxes = []
    confidences = []
    class_ids = []

    x_factor = w / 640.0
    y_factor = h / 640.0

    conf_thresh_float = float(conf_threshold)

    for row in outputs:
        obj_conf = float(row[4])
        if obj_conf >= 0.2:
            classes_scores = row[5:]
            class_id = int(np.argmax(classes_scores))
            max_score = float(classes_scores[class_id])
            conf = float(obj_conf * max_score)

            if conf >= conf_thresh_float and 0 <= class_id < len(COCO_CLASSES):
                cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                left = int((cx - 0.5 * bw) * x_factor)
                top = int((cy - 0.5 * bh) * y_factor)
                width = int(bw * x_factor)
                height = int(bh * y_factor)

                # Boundary safety
                left = max(0, left)
                top = max(0, top)
                width = min(w - left, width)
                height = min(h - top, height)

                boxes.append([left, top, width, height])
                confidences.append(conf)
                class_ids.append(class_id)

    # Non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh_float, float(nms_threshold))

    results_list = []
    class_counts = {}

    for i in indices:
        idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        box = boxes[idx]
        class_name = COCO_CLASSES[class_ids[idx]]
        conf = confidences[idx]
        startX, startY, box_w, box_h = box
        endX, endY = startX + box_w, startY + box_h

        # Distinct color or fallback
        color = CLASS_COLORS.get(class_name, (0, 215, 255))

        # Draw box
        cv2.rectangle(annotated, (startX, startY), (endX, endY), color, 2)

        # Corner HUD accents
        corner_len = min(int(box_w * 0.2), 20)
        if corner_len > 4:
            cv2.line(annotated, (startX, startY), (startX + corner_len, startY), (255, 255, 255), 3)
            cv2.line(annotated, (startX, startY), (startX, startY + corner_len), (255, 255, 255), 3)

        # Label tag
        pct = int(conf * 100)
        label = f"{class_name.upper()} {pct}%"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        tag_y = max(startY, lh + 6)
        cv2.rectangle(annotated, (startX, tag_y - lh - 6), (startX + lw + 10, tag_y + 2), color, -1)
        cv2.putText(annotated, label, (startX + 5, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        results_list.append({
            'class': class_name,
            'confidence': round(conf * 100, 1),
            'box': [int(startX), int(startY), int(box_w), int(box_h)]
        })
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image_safe(annotated, output_path)

    elapsed_ms = (time.time() - start_time) * 1000.0

    meta = {
        'object_count': len(results_list),
        'detections': results_list,
        'summary': class_counts,
        'conf_threshold': conf_threshold,
        'model_name': "YOLOv5 Nano (80 COCO Classes)"
    }
    return True, meta, elapsed_ms


# =====================================================================
# 4. Optional MediaPipe Hands & Pose Detection
# =====================================================================
def detect_mediapipe_landmarks(input_path, output_path, task='hands'):
    """
    Applies MediaPipe Hand Landmark or Pose detection if mediapipe is available.
    """
    start_time = time.time()
    img = read_image_safe(input_path)
    if img is None:
        return False, {"error": "Image could not be read"}, 0.0

    try:
        import mediapipe as mp
    except ImportError:
        return False, {"error": "MediaPipe is not installed in the Python environment"}, 0.0

    annotated = img.copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    count = 0
    if task == 'hands':
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        with mp_hands.Hands(static_image_mode=True, max_num_hands=4, min_detection_confidence=0.5) as hands:
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                count = len(results.multi_hand_landmarks)
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2)
                    )
    elif task == 'pose':
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            results = pose.process(rgb)
            if results.pose_landmarks:
                count = 1
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 128), thickness=2)
                )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image_safe(annotated, output_path)
    elapsed_ms = (time.time() - start_time) * 1000.0

    meta = {
        'task': task,
        'landmarks_detected': count,
        'message': f"Detected {count} {task} instances using MediaPipe"
    }
    return True, meta, elapsed_ms
