"""
AI Vision Studio - Database Layer
Manages SQLite persistence for analysis history, detections, and performance metrics.
"""

import os
import sqlite3
import json
from datetime import datetime

from werkzeug.security import generate_password_hash, check_password_hash

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history.db')


def get_db_connection():
    """Create and return a row-mapped SQLite connection."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the SQLite database tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            feature TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            processed_filename TEXT,
            parameters TEXT,
            details TEXT,
            file_size_kb REAL DEFAULT 0.0,
            processing_time_ms REAL DEFAULT 0.0
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'Visionist',
            avatar_color TEXT DEFAULT '#2563eb',
            created_at TEXT NOT NULL
        )
    ''')

    # Ensure default admin account exists
    cursor.execute('SELECT id FROM users WHERE username = ?', ('admin',))
    if not cursor.fetchone():
        admin_hash = generate_password_hash('admin123')
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, role, avatar_color, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('admin', 'admin@aivision.studio', admin_hash, 'Lead AI Visionist', '#3b82f6', created_at))

    conn.commit()
    conn.close()


def create_user(username, email, password, role='Visionist', avatar_color='#2563eb'):
    """Creates a new user with hashed password. Returns (success, result_or_error)."""
    username = (username or '').strip()
    email = (email or '').strip().lower()

    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters long."
    if not email or '@' not in email:
        return False, "Please provide a valid email address."
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long."

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check for existing user
    cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
    existing = cursor.fetchone()
    if existing:
        conn.close()
        return False, "A user with that username or email already exists."

    pwd_hash = generate_password_hash(password)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, role, avatar_color, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, email, pwd_hash, role, avatar_color, created_at))
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return True, user_id
    except sqlite3.IntegrityError as e:
        conn.close()
        return False, f"Account creation failed: {e}"


def authenticate_user(username_or_email, password):
    """Verifies credentials. Returns user dict on success, None on failure."""
    if not username_or_email or not password:
        return None

    identifier = username_or_email.strip()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM users 
        WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)
    ''', (identifier, identifier))
    row = cursor.fetchone()
    conn.close()

    if row and check_password_hash(row['password_hash'], password):
        user = dict(row)
        user.pop('password_hash', None)
        return user
    return None


def get_user_by_id(user_id):
    """Retrieves user profile by ID without password hash."""
    if not user_id:
        return None
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, email, role, avatar_color, created_at FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_username(username):
    """Retrieves user profile by username."""
    if not username:
        return None
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, email, role, avatar_color, created_at FROM users WHERE LOWER(username) = LOWER(?)', (username.strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def add_record(feature, original_filename, processed_filename=None,
               parameters=None, details=None, file_size_kb=0.0, processing_time_ms=0.0):
    """
    Inserts a new record into analysis_history.
    parameters and details can be dicts or strings (will be serialized if dict).
    """
    if isinstance(parameters, dict):
        parameters = json.dumps(parameters)
    if isinstance(details, (dict, list)):
        details = json.dumps(details)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO analysis_history 
        (timestamp, feature, original_filename, processed_filename, parameters, details, file_size_kb, processing_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, feature, original_filename, processed_filename, parameters, details, round(file_size_kb, 2), round(processing_time_ms, 2)))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_recent_records(limit=50, feature=None):
    """Retrieves recent history records, optionally filtered by feature."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if feature and feature != 'all':
        cursor.execute('''
            SELECT * FROM analysis_history 
            WHERE feature = ? 
            ORDER BY id DESC LIMIT ?
        ''', (feature, limit))
    else:
        cursor.execute('''
            SELECT * FROM analysis_history 
            ORDER BY id DESC LIMIT ?
        ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_record_by_id(record_id):
    """Retrieves a single record by its ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM analysis_history WHERE id = ?', (record_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_stats():
    """Calculates summary statistics for the dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM analysis_history')
    total_analyses = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE feature = 'image_processing'")
    image_processing_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE feature = 'face_detection'")
    face_count_runs = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE feature = 'object_detection'")
    object_count_runs = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE feature = 'webcam_capture'")
    webcam_count_runs = cursor.fetchone()[0]

    # Calculate total faces detected from details column
    cursor.execute("SELECT details FROM analysis_history WHERE feature = 'face_detection'")
    face_rows = cursor.fetchall()
    total_faces_found = 0
    for r in face_rows:
        if r['details']:
            try:
                data = json.loads(r['details'])
                if isinstance(data, dict):
                    total_faces_found += data.get('face_count', 0)
            except Exception:
                pass

    # Calculate total objects detected
    cursor.execute("SELECT details FROM analysis_history WHERE feature = 'object_detection'")
    obj_rows = cursor.fetchall()
    total_objects_found = 0
    for r in obj_rows:
        if r['details']:
            try:
                data = json.loads(r['details'])
                if isinstance(data, dict):
                    total_objects_found += data.get('object_count', 0)
            except Exception:
                pass

    conn.close()
    return {
        'total_analyses': total_analyses,
        'image_processing_count': image_processing_count,
        'face_detection_count': face_count_runs,
        'object_detection_count': object_count_runs,
        'webcam_count': webcam_count_runs,
        'total_faces_found': total_faces_found,
        'total_objects_found': total_objects_found
    }


def delete_record(record_id):
    """Deletes a single history record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM analysis_history WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()


def clear_all_records():
    """Wipes all history records."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM analysis_history')
    conn.commit()
    conn.close()
