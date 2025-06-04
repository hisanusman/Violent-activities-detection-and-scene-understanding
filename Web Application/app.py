from flask import Flask, render_template, Response, redirect, url_for
import sqlite3
import os
import cv2
from datetime import timedelta, datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Prediction Script'))
from predictor import process_frame  # This function now uses an extra check to avoid duplicate timestamps.
from utils import generate_scene_description, generate_pdf_report, class_labels
from threading import Lock
import time

app = Flask(__name__)

# Update the list: use "droidcam" placeholder for the live feed (index 2)
VIDEOS = [
    "cam0.mp4",
    "cam1.mp4",
    "droidcam",   # This entry now refers to the live DroidCam feed.
    "cam3.mp4",
]

# Global dictionaries to track processing state per video_id.
processing_stop_flags = {}      # e.g., { video_id: True/False }
processing_resume_indices = {}  # e.g., { video_id: last_frame_index } to resume from.
processing_summaries = {}       # e.g., { video_id: summary_data }

# Create necessary directories
PREDICTION_SCRIPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'Prediction Script')
TIMESTAMPS_DIR = os.path.join(PREDICTION_SCRIPT_DIR, 'Timestamps')
os.makedirs(TIMESTAMPS_DIR, exist_ok=True)

# Create static directories
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
VIDEOS_DIR = os.path.join(STATIC_DIR, 'videos')
PROCESSED_DIR = os.path.join(STATIC_DIR, 'processed')
PDFS_DIR = os.path.join(STATIC_DIR, 'pdfs')

for directory in [STATIC_DIR, VIDEOS_DIR, PROCESSED_DIR, PDFS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ------------------ Helper DB Functions ------------------
def fetch_timestamps(video_name):
    db_path = os.path.join(TIMESTAMPS_DIR, 'detection_timestamps.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = "SELECT timestamp FROM timestamps WHERE video_name = ?"
    cursor.execute(query, (video_name,))
    timestamps = cursor.fetchall()
    conn.close()
    return [t[0] for t in timestamps]

def fetch_detection_counts(video_name):
    db_path = os.path.join(TIMESTAMPS_DIR, 'Report_Requirments.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = "SELECT video_name, activity FROM detection_counts WHERE video_name = ?"
    cursor.execute(query, (video_name,))
    counts = cursor.fetchall()
    conn.close()
    return counts

def upsert_detection_counts(video_name, most_frequent_activity, unique_weapon_count):
    """Upsert the detection_counts table for a given video_name."""
    db_path = os.path.join(TIMESTAMPS_DIR, 'Report_Requirments.db')
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT UNIQUE,
            activity TEXT,
            count_weapon INTEGER
        )
    ''')
    conn.commit()
    cursor.execute("SELECT id FROM detection_counts WHERE video_name = ?", (video_name,))
    row = cursor.fetchone()
    if row:
        cursor.execute('''
            UPDATE detection_counts 
            SET activity = ?, count_weapon = ?
            WHERE video_name = ?
        ''', (most_frequent_activity, unique_weapon_count, video_name))
    else:
        cursor.execute('''
            INSERT INTO detection_counts (video_name, activity, count_weapon)
            VALUES (?, ?, ?, ?)
        ''', (video_name, most_frequent_activity, unique_weapon_count))
    conn.commit()
    conn.close()

# ------------------ Routes ------------------

@app.route('/')
def index():
    return render_template('index.html', videos=VIDEOS)

# When a video is clicked from the grid, the video page (big screen) is shown.
# Processing starts (or resumes) automatically.
@app.route('/video/<int:video_id>')
def video(video_id):
    if 0 <= video_id < len(VIDEOS):
        processing_stop_flags[video_id] = False  # Clear stop flag so processing will run.
        return render_template('video.html', video=VIDEOS[video_id], video_id=video_id)
    return "Video not found", 404

# Navigator route: stops processing and then displays the processed video file.
@app.route('/Navigation/<int:video_id>')
def navigation(video_id):
    if 0 <= video_id < len(VIDEOS):
        processing_stop_flags[video_id] = True
        time.sleep(1)  # Allow generator to finish
        video_name = VIDEOS[video_id]
        # For the live feed, assign a unique identifier.
        count_vid_name = "droidcam_live" if video_id == 2 else video_name.rsplit('.', 1)[0]
        processed_video_name = f"{count_vid_name}_processed.webm"
        times = fetch_timestamps(count_vid_name)
        return render_template('Navigation.html', video=processed_video_name, video_id=video_id, times=times)
    return "Video not found", 404

# Create Report route stops processing and then generates a report.
@app.route('/create_report/<int:video_id>')
def report(video_id):
    if 0 <= video_id < len(VIDEOS):
        processing_stop_flags[video_id] = True
        time.sleep(1)  # Allow generator to finish.
        video_name = VIDEOS[video_id]
        count_vid_name = "droidcam_live" if video_id == 2 else video_name.rsplit('.', 1)[0]
        summary_data = processing_summaries.get(video_id)
        
        # Get the most frequent activity
        if summary_data is None:
            counts = fetch_detection_counts(count_vid_name)
            if counts:
                _, most_frequent_activity = counts[0]  # Now only expecting 2 values
            else:
                return "No detection summary available.", 404
        else:
            most_frequent_activity = max(summary_data['activity_counts'], key=summary_data['activity_counts'].get)
        
        # Generate PDF report directly
        output_pdf_path = os.path.join(PDFS_DIR, f"{count_vid_name}.pdf")
        try:
            scene_description = generate_scene_description(most_frequent_activity)
            generate_pdf_report(most_frequent_activity, scene_description, output_pdf_path)
        except Exception as e:
            print("Error generating PDF:", e)
            return "Error generating report", 500
            
        rel_pdf_path = f"pdfs/{count_vid_name}.pdf"
        return render_template('Report.html', 
                             video=count_vid_name, 
                             video_id=video_id, 
                             activity=most_frequent_activity,
                             scene_description=scene_description,
                             pdf_path=rel_pdf_path)
    return "Video not found", 404

# ------------------ Real-Time Streaming (with saving to file) ------------------
def try_connect_droidcam(max_retries=3):
    """Try to connect to DroidCam with multiple URLs and retries."""
    urls = [
        'http://192.168.18.16:4747/video',  # Current URL
        'http://192.168.18.93:4747/video',  # Previous URL
        'http://127.0.0.1:4747/video'       # Localhost fallback
    ]
    
    for url in urls:
        for attempt in range(max_retries):
            try:
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"Successfully connected to DroidCam at {url}")
                        return cap
                    cap.release()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
            time.sleep(1)  # Wait before retry
    
    print("Failed to connect to DroidCam on all URLs")
    return None

def generate_real_time_frames(video_source, video_id):
    # If video_id == 2, use the DroidCam live feed.
    if video_id == 2:
        time.sleep(3)  # Wait for DroidCam to start streaming.
        cap = try_connect_droidcam()
        if cap is None:
            print("Error: Could not connect to DroidCam")
            return
        fps = 50  # Set a default FPS for the live feed.
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error opening video file for realtime processing.")
            return
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    frame_index = 0

    # Determine processed video file output path.
    video_name = "droidcam_live" if video_id == 2 else os.path.splitext(os.path.basename(video_source))[0]
    processed_dir = os.path.join(app.static_folder, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    output_video_path = os.path.join(processed_dir, f"{video_name}_processed.webm")

    # For file-based video, get dimensions from the capture.
    if video_id != 2:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        # For DroidCam, try to get dimensions, fallback to defaults if needed
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # For live feed, resume functionality is not applicable.
    if video_id != 2:
        resume_index = processing_resume_indices.get(video_id)
        if resume_index is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, resume_index)
            frame_index = resume_index

    # Setup DB connection for timestamps.
    ts_db_path = os.path.join(TIMESTAMPS_DIR, 'detection_timestamps.db')
    conn = sqlite3.connect(ts_db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS timestamps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()

    # Initialize realtime summary data and locks.
    summary_data = {
        'unique_persons': [],
        'unique_weapons': [],
        'activity_counts': {label: 0 for label in class_labels.values()}
    }
    if video_id == 2:  # For droidcam, initialize with only "normal" activity
        summary_data['activity_counts'] = {label: 0 for label in class_labels.values()}
        summary_data['activity_counts']["normal"] = 1  # Set initial count for normal

    db_lock = Lock()
    summary_lock = Lock()
    last_state = {'last_activity': None, 'last_inserted_frame': -1}
    
    while True:
        if processing_stop_flags.get(video_id, False):
            if video_id != 2:
                processing_resume_indices[video_id] = frame_index
            print(f"Processing stopped for video_id {video_id} at frame {frame_index}")
            break

        ret, frame = cap.read()
        if not ret:
            # For live feed, try to reconnect if connection is lost
            if video_id == 2:
                print("Lost connection to DroidCam, attempting to reconnect...")
                cap.release()
                cap = try_connect_droidcam()
                if cap is None:
                    print("Failed to reconnect to DroidCam")
                    break
                continue
            else:
                break

        frame_index += 1
        # Process frame: this function handles detection, DB updates, and summary data.
        processed_frame = process_frame(frame, frame_index, fps, video_name, cursor, db_lock, summary_data, summary_lock, last_state)
        out_writer.write(processed_frame)  # Save processed frame to file

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    out_writer.release()
    conn.close()

    # For droidcam, always set the most frequent activity as "normal"
    if video_id == 2:
        most_frequent_activity = "normal"
    else:
        most_frequent_activity = max(summary_data['activity_counts'], key=summary_data['activity_counts'].get)
    
    # Update the detection counts in the database
    upsert_detection_counts(video_name, most_frequent_activity, len(summary_data['unique_weapons']))
    processing_summaries[video_id] = summary_data
    print("Final summary updated for:", video_name)

def generate_raw_frames():
    """Generate raw frames from DroidCam without processing."""
    time.sleep(3)  # Wait for DroidCam to start streaming.
    cap = try_connect_droidcam()
    if cap is None:
        print("Error: Could not connect to DroidCam for raw stream")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection to DroidCam, attempting to reconnect...")
            cap.release()
            cap = try_connect_droidcam()
            if cap is None:
                print("Failed to reconnect to DroidCam")
                break
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/raw_stream/<int:video_id>')
def raw_stream(video_id):
    """Serves raw live feed only for DroidCam (video_id 2)."""
    if video_id == 2:
        return Response(generate_raw_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Not available", 404

# Route to stream processed frames.
@app.route('/stream_video/<int:video_id>')
def stream_video(video_id):
    if 0 <= video_id < len(VIDEOS):
        video_source = "droidcam_live" if video_id == 2 else os.path.join(app.static_folder, "videos", VIDEOS[video_id])
        return Response(generate_real_time_frames(video_source, video_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Video not found", 404

# ------------------ Run App ------------------
if __name__ == '__main__':
    app.run(debug=True)
