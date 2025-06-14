from flask import Flask, render_template, Response, redirect, url_for
import sqlite3
import os
import cv2
from datetime import timedelta, datetime
import sys
sys.path.append(os.path.abspath("D:\\FYP\\FYP\\Violent-activities-detection-and-scene-understanding\\Prediction Script\\"))
from predictor import process_frame # This function now uses an extra check to avoid duplicate timestamps.
from utils import generate_scene_description, generate_crime_scene_report_pdf, class_labels
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

# ------------------ Helper DB Functions ------------------
def fetch_timestamps(video_name):
    conn = sqlite3.connect('D:\\FYP\\FYP\\Violent-activities-detection-and-scene-understanding\\Prediction Script\\Timestamps\\detection_timestamps.db')
    cursor = conn.cursor()
    query = "SELECT timestamp FROM timestamps WHERE video_name = ?"
    cursor.execute(query, (video_name,))
    timestamps = cursor.fetchall()
    conn.close()
    return [t[0] for t in timestamps]

def fetch_detection_counts(video_name):
    conn = sqlite3.connect('D:\\FYP\\FYP\\Violent-activities-detection-and-scene-understanding\\Prediction Script\\Timestamps\\Report_Requirments.db')
    cursor = conn.cursor()
    query = "SELECT video_name, activity, count_person, count_weapon FROM detection_counts WHERE video_name = ?"
    cursor.execute(query, (video_name,))
    counts = cursor.fetchall()
    conn.close()
    return counts

def upsert_detection_counts(video_name, most_frequent_activity, unique_person_count, unique_weapon_count):
    """Upsert the detection_counts table for a given video_name."""
    report_db_path = 'D:\\FYP\\FYP\\Violent-activities-detection-and-scene-understanding\\Prediction Script\\Timestamps\\Report_Requirments.db'
    conn = sqlite3.connect(report_db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT UNIQUE,
            activity TEXT,
            count_person INTEGER,
            count_weapon INTEGER
        )
    ''')
    conn.commit()
    cursor.execute("SELECT id FROM detection_counts WHERE video_name = ?", (video_name,))
    row = cursor.fetchone()
    if row:
        cursor.execute('''
            UPDATE detection_counts 
            SET activity = ?, count_person = ?, count_weapon = ?
            WHERE video_name = ?
        ''', (most_frequent_activity, unique_person_count, unique_weapon_count, video_name))
    else:
        cursor.execute('''
            INSERT INTO detection_counts (video_name, activity, count_person, count_weapon)
            VALUES (?, ?, ?, ?)
        ''', (video_name, most_frequent_activity, unique_person_count, unique_weapon_count))
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
        if summary_data is None:
            counts = fetch_detection_counts(count_vid_name)
            if counts:
                _, most_frequent_activity, unique_person_count, unique_weapon_count = counts[0]
            else:
                return "No detection summary available.", 404
        else:
            most_frequent_activity = max(summary_data['activity_counts'], key=summary_data['activity_counts'].get)
            unique_person_count = len(summary_data['unique_persons'])
            unique_weapon_count = len(summary_data['unique_weapons'])
            upsert_detection_counts(count_vid_name, most_frequent_activity, unique_person_count, unique_weapon_count)
        scene_description = generate_scene_description(most_frequent_activity, unique_person_count, unique_weapon_count)
        output_dir = os.path.join("D:\\FYP\\FYP\\Violent-activities-detection-and-scene-understanding\\Web Application\\static", "pdfs")
        os.makedirs(output_dir, exist_ok=True)
        output_pdf_path = os.path.join(output_dir, f"{count_vid_name}.pdf")
        try:
            generate_crime_scene_report_pdf(most_frequent_activity, scene_description, unique_person_count, unique_weapon_count, output_pdf_path)
        except Exception as e:
            print("Error generating PDF:", e)
        rel_pdf_path = f"pdfs/{count_vid_name}.pdf"
        return render_template('Report.html', 
                               video=count_vid_name, 
                               video_id=video_id, 
                               activity=most_frequent_activity, 
                               unique_person_count=unique_person_count, 
                               unique_weapon_count=unique_weapon_count, 
                               scene_description=scene_description,
                               pdf_path=rel_pdf_path)
    return "Video not found", 404

# ------------------ Real-Time Streaming (with saving to file) ------------------
def generate_real_time_frames(video_source, video_id):
    # If video_id == 2, use the DroidCam live feed.
    if video_id == 2:
        # Replace with your DroidCam URL (e.g., the IP of your DroidCam phone and port).
        time.sleep(3)  # Wait for DroidCam to start streaming.
        cap = cv2.VideoCapture('http://192.168.100.33:4747/video')
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
        # Set dimensions manually for the DroidCam stream if not auto-detected.
        frame_width, frame_height = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # For live feed, resume functionality is not applicable.
    if video_id != 2:
        resume_index = processing_resume_indices.get(video_id)
        if resume_index is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, resume_index)
            frame_index = resume_index

    # Setup DB connection for timestamps.
    ts_db_path = 'D:\\FYP\\FYP\\Violent-activities-detection-and-scene-understanding\\Prediction Script\\Timestamps\\detection_timestamps.db'
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
    db_lock = Lock()
    summary_lock = Lock()
    # Extend last_state to include last inserted frame index to avoid duplicates.
    last_state = {'last_activity': None, 'last_inserted_frame': -1}
    
    while True:
        if processing_stop_flags.get(video_id, False):
            if video_id != 2:
                processing_resume_indices[video_id] = frame_index
            print(f"Processing stopped for video_id {video_id} at frame {frame_index}")
            break

        ret, frame = cap.read()
        if not ret:
            # For live feed, if a frame fails, continue to try reading new frames.
            if video_id == 2:
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

    most_frequent_activity = max(summary_data['activity_counts'], key=summary_data['activity_counts'].get)
    upsert_detection_counts(video_name, most_frequent_activity, len(summary_data['unique_persons']), len(summary_data['unique_weapons']))
    processing_summaries[video_id] = summary_data
    print("Final summary updated for:", video_name)

def generate_raw_frames():
    """Generate raw frames from DroidCam without processing."""
    time.sleep(3)  # Wait for DroidCam to start streaming.
    cap = cv2.VideoCapture('http://192.168.100.33:4747/video')
    while True:
        ret, frame = cap.read()
        if not ret:
            #print("No frame received from DroidCam. Retrying...")
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
