"""
AI Vision Studio - Flask Application Server
Main entry point handling routing, file uploads, REST API endpoints,
and integration with vision algorithms and SQLite database.
"""

import os
import uuid
import base64
import time
from functools import wraps
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

import database
import vision

# Application configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(STATIC_DIR, 'results')
SAMPLES_FOLDER = os.path.join(STATIC_DIR, 'samples')

app = Flask(__name__)
app.secret_key = 'ai-vision-studio-secret-key-2026'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['SAMPLES_FOLDER'] = SAMPLES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload limit

# Ensure directories exist
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, SAMPLES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize database
database.init_db()


def save_uploaded_or_sample(req_files, sample_name=None):
    """
    Saves an uploaded file or resolves a sample file.
    Returns: (success, original_filename, disk_path, relative_url)
    """
    if 'image' in req_files and req_files['image'].filename != '':
        file = req_files['image']
        if not vision.allowed_file(file.filename):
            return False, "Unsupported file format. Please upload JPG, PNG, WEBP, or BMP.", None, None

        orig_name = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex[:8]}_{orig_name}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(save_path)
        rel_url = f"/static/uploads/{unique_name}"
        return True, orig_name, save_path, rel_url

    # Check if a sample was specified
    if sample_name:
        sample_path = os.path.join(app.config['SAMPLES_FOLDER'], secure_filename(sample_name))
        if os.path.exists(sample_path):
            unique_name = f"sample_{uuid.uuid4().hex[:8]}_{secure_filename(sample_name)}"
            dest_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            # Copy sample to uploads
            with open(sample_path, 'rb') as src, open(dest_path, 'wb') as dst:
                dst.write(src.read())
            rel_url = f"/static/uploads/{unique_name}"
            return True, sample_name, dest_path, rel_url

    return False, "No image uploaded or sample selected", None, None


# =====================================================================
# Authentication & User Session Management
# =====================================================================

@app.context_processor
def inject_current_user():
    """Provides current_user to all Jinja2 templates."""
    user = None
    if 'user_id' in session:
        user = database.get_user_by_id(session['user_id'])
        if not user:
            session.pop('user_id', None)
            session.pop('username', None)
    return {'current_user': user}


def login_required(f):
    """Decorator to require login for specific routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this studio section.', 'info')
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User entry / login page with one-click demo login."""
    if 'user_id' in session:
        flash(f"You are already signed in as {session.get('username')}.", 'info')
        return redirect(url_for('index'))

    next_page = request.args.get('next') or request.form.get('next') or url_for('index')

    if request.method == 'POST':
        action = request.form.get('action')

        # One-click demo login
        if action == 'demo_login':
            user = database.authenticate_user('admin', 'admin123')
            if user:
                session['user_id'] = user['id']
                session['username'] = user['username']
                flash(f"Welcome to AI Vision Studio, {user['username']}! Logged in with Lead Visionist demo credentials.", 'success')
                return redirect(next_page)

        identifier = request.form.get('identifier', '').strip()
        password = request.form.get('password', '')

        if not identifier or not password:
            flash("Please enter both username/email and your password.", "warning")
            return render_template('login.html', next=next_page, identifier=identifier)

        user = database.authenticate_user(identifier, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f"Welcome back, {user['username']}!", 'success')
            return redirect(next_page)
        else:
            flash("Invalid username/email or password. Please try again.", 'error')
            return render_template('login.html', next=next_page, identifier=identifier)

    return render_template('login.html', next=next_page)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page."""
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        role = request.form.get('role', 'AI Visionist')

        if password != confirm_password:
            flash("Passwords do not match. Please re-enter them.", 'error')
            return render_template('register.html', username=username, email=email, role=role)

        palette = ['#2563eb', '#7c3aed', '#059669', '#d97706', '#dc2626', '#0891b2']
        avatar_color = palette[len(username) % len(palette)]

        success, result = database.create_user(username, email, password, role=role, avatar_color=avatar_color)
        if success:
            session['user_id'] = result
            session['username'] = username
            flash(f"Account created successfully! Welcome aboard, {username}.", 'success')
            return redirect(url_for('index'))
        else:
            flash(result, 'error')
            return render_template('register.html', username=username, email=email, role=role)

    return render_template('register.html')


@app.route('/logout')
def logout():
    """Logs out current user session."""
    user = session.get('username', 'Guest')
    session.clear()
    flash(f"Goodbye {user}, you have been signed out safely.", 'info')
    return redirect(url_for('login'))


# =====================================================================
# Pages
# =====================================================================

@app.route('/')
def index():
    """Dashboard homepage with stats and feature navigation."""
    stats = database.get_stats()
    recent = database.get_recent_records(limit=6)
    return render_template('index.html', stats=stats, recent_activities=recent)


@app.route('/image-processing')
def image_processing():
    """Image processing playground page."""
    samples = [f for f in os.listdir(app.config['SAMPLES_FOLDER']) if vision.allowed_file(f)]
    return render_template('image_processing.html', samples=samples)


@app.route('/face-detection')
def face_detection():
    """Haar Cascade face detection page."""
    samples = [f for f in os.listdir(app.config['SAMPLES_FOLDER']) if vision.allowed_file(f)]
    return render_template('face_detection.html', samples=samples)


@app.route('/webcam')
def webcam():
    """Live webcam studio page."""
    return render_template('webcam.html')


@app.route('/object-detection')
def object_detection():
    """Pretrained Object Detection page."""
    samples = [f for f in os.listdir(app.config['SAMPLES_FOLDER']) if vision.allowed_file(f)]
    return render_template('object_detection.html', samples=samples)


@app.route('/history')
def history():
    """Analysis history and logs page."""
    records = database.get_recent_records(limit=100)
    stats = database.get_stats()
    return render_template('history.html', records=records, stats=stats)


# =====================================================================
# API Endpoints
# =====================================================================

@app.route('/api/process-image', methods=['POST'])
def api_process_image():
    """Handles image processing filter requests."""
    sample_name = request.form.get('sample_name')
    success, orig_name, input_path, orig_url = save_uploaded_or_sample(request.files, sample_name)
    if not success:
        return jsonify({'success': False, 'error': orig_name}), 400

    params = {
        'color_mode': request.form.get('color_mode', 'original'),
        'brightness': float(request.form.get('brightness', 0)),
        'contrast': float(request.form.get('contrast', 1.0)),
        'blur_type': request.form.get('blur_type', 'none'),
        'blur_intensity': int(request.form.get('blur_intensity', 0)),
        'edge_type': request.form.get('edge_type', 'none'),
        'canny_low': int(request.form.get('canny_low', 50)),
        'canny_high': int(request.form.get('canny_high', 150)),
        'threshold_type': request.form.get('threshold_type', 'none'),
        'threshold_val': int(request.form.get('threshold_val', 127)),
        'resize_mode': request.form.get('resize_mode', 'none'),
        'scale_percent': int(request.form.get('scale_percent', 100)),
        'target_width': request.form.get('target_width'),
        'target_height': request.form.get('target_height'),
        'rotation': int(request.form.get('rotation', 0)),
        'flip': request.form.get('flip', 'none')
    }

    output_filename = f"processed_{uuid.uuid4().hex[:8]}_{orig_name}"
    output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)

    proc_ok, meta, elapsed_ms = vision.apply_image_playground_filters(input_path, output_path, params)
    if not proc_ok:
        return jsonify({'success': False, 'error': meta.get('error', 'Processing failed')}), 500

    processed_url = f"/static/results/{output_filename}"
    file_size_kb = meta.get('size_kb', 0.0)

    # Save to history
    database.add_record(
        feature='image_processing',
        original_filename=orig_name,
        processed_filename=output_filename,
        parameters=params,
        details=meta.get('filters_applied', {}),
        file_size_kb=file_size_kb,
        processing_time_ms=elapsed_ms
    )

    orig_meta = vision.get_image_metadata(input_path)

    return jsonify({
        'success': True,
        'original_url': orig_url,
        'processed_url': processed_url,
        'output_filename': output_filename,
        'metadata': meta,
        'original_metadata': orig_meta,
        'elapsed_ms': round(elapsed_ms, 2)
    })


@app.route('/api/detect-faces', methods=['POST'])
def api_detect_faces():
    """Handles Haar Cascade Face Detection requests."""
    sample_name = request.form.get('sample_name')
    success, orig_name, input_path, orig_url = save_uploaded_or_sample(request.files, sample_name)
    if not success:
        return jsonify({'success': False, 'error': orig_name}), 400

    scale_factor = float(request.form.get('scale_factor', 1.1))
    min_neighbors = int(request.form.get('min_neighbors', 5))
    min_size = int(request.form.get('min_size', 30))
    detect_eyes = request.form.get('detect_eyes') in ['true', '1', 'on']

    output_filename = f"face_{uuid.uuid4().hex[:8]}_{orig_name}"
    output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)

    proc_ok, meta, elapsed_ms = vision.detect_faces(
        input_path, output_path,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=min_size,
        detect_eyes=detect_eyes
    )

    if not proc_ok:
        return jsonify({'success': False, 'error': meta.get('error', 'Face detection failed')}), 500

    processed_url = f"/static/results/{output_filename}"
    file_size_kb = round(os.path.getsize(output_path) / 1024, 2) if os.path.exists(output_path) else 0.0

    # Save to database
    database.add_record(
        feature='face_detection',
        original_filename=orig_name,
        processed_filename=output_filename,
        parameters={'scale_factor': scale_factor, 'min_neighbors': min_neighbors, 'detect_eyes': detect_eyes},
        details=meta,
        file_size_kb=file_size_kb,
        processing_time_ms=elapsed_ms
    )

    return jsonify({
        'success': True,
        'original_url': orig_url,
        'processed_url': processed_url,
        'output_filename': output_filename,
        'face_count': meta.get('face_count', 0),
        'faces': meta.get('faces', []),
        'note': meta.get('note'),
        'elapsed_ms': round(elapsed_ms, 2)
    })


@app.route('/api/detect-objects', methods=['POST'])
def api_detect_objects():
    """Handles Pretrained Object Detection requests (OpenCV DNN MobileNet-SSD)."""
    sample_name = request.form.get('sample_name')
    success, orig_name, input_path, orig_url = save_uploaded_or_sample(request.files, sample_name)
    if not success:
        return jsonify({'success': False, 'error': orig_name}), 400

    conf_threshold = float(request.form.get('conf_threshold', 0.45))
    output_filename = f"obj_{uuid.uuid4().hex[:8]}_{orig_name}"
    output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)

    proc_ok, meta, elapsed_ms = vision.detect_objects_dnn(input_path, output_path, conf_threshold=conf_threshold)
    if not proc_ok:
        return jsonify({'success': False, 'error': meta.get('error', 'Object detection failed')}), 500

    processed_url = f"/static/results/{output_filename}"
    file_size_kb = round(os.path.getsize(output_path) / 1024, 2) if os.path.exists(output_path) else 0.0

    # Save to database
    database.add_record(
        feature='object_detection',
        original_filename=orig_name,
        processed_filename=output_filename,
        parameters={'conf_threshold': conf_threshold},
        details=meta,
        file_size_kb=file_size_kb,
        processing_time_ms=elapsed_ms
    )

    return jsonify({
        'success': True,
        'original_url': orig_url,
        'processed_url': processed_url,
        'output_filename': output_filename,
        'object_count': meta.get('object_count', 0),
        'detections': meta.get('detections', []),
        'summary': meta.get('summary', {}),
        'elapsed_ms': round(elapsed_ms, 2)
    })


@app.route('/api/webcam-capture', methods=['POST'])
def api_webcam_capture():
    """Handles snapshot uploaded directly from browser webcam."""
    data = request.get_json()
    if not data or 'image_data' not in data:
        return jsonify({'success': False, 'error': 'No image data received'}), 400

    raw_data = data['image_data']
    filter_mode = data.get('filter_mode', 'original')

    # Parse base64
    if ',' in raw_data:
        raw_data = raw_data.split(',', 1)[1]

    try:
        image_bytes = base64.b64decode(raw_data)
    except Exception as e:
        return jsonify({'success': False, 'error': 'Malformed base64 image data'}), 400

    # Save initial snapshot
    orig_name = f"webcam_{int(time.time())}.jpg"
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], orig_name)
    with open(input_path, 'wb') as f:
        f.write(image_bytes)

    orig_url = f"/static/uploads/{orig_name}"
    output_filename = f"webcam_proc_{int(time.time())}.jpg"
    output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)

    meta = {}
    elapsed_ms = 0.0

    if filter_mode == 'face':
        proc_ok, meta, elapsed_ms = vision.detect_faces(input_path, output_path, scale_factor=1.1, min_neighbors=5)
    elif filter_mode == 'object':
        proc_ok, meta, elapsed_ms = vision.detect_objects_dnn(input_path, output_path, conf_threshold=0.4)
    elif filter_mode in ['grayscale', 'blur', 'canny', 'sepia', 'invert', 'sharpen', 'sketch']:
        params = {
            'color_mode': filter_mode if filter_mode in ['grayscale', 'sepia', 'invert', 'sharpen', 'sketch'] else 'original',
            'blur_type': 'gaussian' if filter_mode == 'blur' else 'none',
            'blur_intensity': 15 if filter_mode == 'blur' else 0,
            'edge_type': 'canny' if filter_mode == 'canny' else 'none',
            'canny_low': 50,
            'canny_high': 150
        }
        proc_ok, meta, elapsed_ms = vision.apply_image_playground_filters(input_path, output_path, params)
    else:
        # Just copy as original
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        proc_ok = True
        elapsed_ms = 5.0

    processed_url = f"/static/results/{output_filename}" if proc_ok else orig_url
    file_size_kb = round(os.path.getsize(output_path) / 1024, 2) if os.path.exists(output_path) else 0.0

    # Save to history
    database.add_record(
        feature='webcam_capture',
        original_filename=orig_name,
        processed_filename=output_filename if proc_ok else orig_name,
        parameters={'filter_mode': filter_mode},
        details=meta,
        file_size_kb=file_size_kb,
        processing_time_ms=elapsed_ms
    )

    return jsonify({
        'success': True,
        'original_url': orig_url,
        'processed_url': processed_url,
        'output_filename': output_filename,
        'filter_mode': filter_mode,
        'meta': meta,
        'elapsed_ms': round(elapsed_ms, 2)
    })


@app.route('/download/<folder>/<filename>')
def download_file(folder, filename):
    """Secure file download route."""
    safe_name = secure_filename(filename)
    if folder == 'results':
        dir_path = app.config['RESULTS_FOLDER']
    elif folder == 'uploads':
        dir_path = app.config['UPLOAD_FOLDER']
    else:
        return "Invalid directory", 400

    return send_from_directory(dir_path, safe_name, as_attachment=True)


@app.route('/api/history/delete/<int:record_id>', methods=['POST'])
def api_delete_history(record_id):
    """Deletes a history record."""
    database.delete_record(record_id)
    return jsonify({'success': True, 'record_id': record_id})


@app.route('/api/history/clear', methods=['POST'])
def api_clear_history():
    """Clears all history records."""
    database.clear_all_records()
    return jsonify({'success': True})


if __name__ == '__main__':
    print("==================================================")
    print(">> AI Vision Studio is launching on http://localhost:5000 (http://127.0.0.1:5000)")
    print("==================================================")
    app.run(debug=True, host='0.0.0.0', port=5000)
