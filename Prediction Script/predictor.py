import cv2
import os
import sqlite3
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from utils import (
    predict_activity, detect_objects, update_unique_objects,
    identify_weapon_holder, run_pose_estimation_and_save, class_labels
)
import warnings
warnings.filterwarnings("ignore")  # Suppress all Python warnings

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow warnings
def process_frame(frame, frame_index, fps, video_name, cursor, db_lock, summary_data, summary_lock, last_state):
    """
    Processes a single frame:
      - Computes elapsed time.
      - Runs activity detection and object detection (GPU accelerated).
      - Updates summary counters and the timestamps database if an event is detected.
      - Draws bounding boxes, overlays text, and overlays the cropped weapon holder (if any).
    """
    elapsed_time = frame_index / fps
    elapsed_time_str = str(timedelta(seconds=elapsed_time))
    
    # GPU inference: the models in utils are assumed loaded onto GPU (if available)
    detected_activity = predict_activity(frame)
    weapons, persons = detect_objects(frame)
    
    # Process weapon holder detection and crop extraction
    weapon_holder = identify_weapon_holder(weapons, persons)
    weapon_holder_crop = None
    frame_height, frame_width = frame.shape[:2]
    if weapon_holder is not None:
        px1, py1, px2, py2 = weapon_holder
        px1, py1 = max(px1, 0), max(py1, 0)
        px2, py2 = min(px2, frame_width), min(py2, frame_height)
        crop_img = frame[py1:py2, px1:px2].copy()
        weapon_holder_crop = run_pose_estimation_and_save(crop_img, frame_index)
    
    # Update shared summary data (unique persons, weapons, and activity counts)
    with summary_lock:
        update_unique_objects(summary_data['unique_persons'], persons, threshold=50)
        update_unique_objects(summary_data['unique_weapons'], weapons, threshold=50)
        summary_data['activity_counts'][detected_activity] += 1
    
    # Update the timestamps DB if activity changes or a weapon holder is detected
    if detected_activity != last_state.get('last_activity') or weapon_holder is not None:
        with db_lock:
            cursor.execute(
                'INSERT INTO timestamps (video_name, timestamp) VALUES (?, ?)',
                (video_name, elapsed_time_str)
            )
            cursor.connection.commit()
        last_state['last_activity'] = detected_activity
    
    # Draw bounding boxes for persons and weapons
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for (x1, y1, x2, y2) in weapons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if weapon_holder is not None:
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
    
    # Overlay activity and timestamp text
    cv2.putText(frame, f"Activity: {detected_activity}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Time: {elapsed_time_str}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Overlay the weapon holder crop if available
    if weapon_holder_crop is not None:
        overlay = cv2.resize(weapon_holder_crop, (150, 150))
        frame[0:150, frame_width-150:frame_width] = overlay

    return frame

def run_detection_multithread(video_path, output_video_path):
    """
    Processes the input video frame-by-frame using multithreading,
    leveraging GPU acceleration for inference.
    In addition to writing processed frames, it logs timestamps to a SQLite
    database and aggregates summary counts which are stored in a separate report DB.
    """
    # Set up the timestamps database (shared among threads)
    ts_db_path = 'D:\\FYP\\Violent-activities-detection-and-scene-understanding\\Prediction Script\\Timestamps\\detection_timestamps.db'
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
    
    # Create a global lock for DB operations on the timestamps DB.
    db_lock = Lock()
    
    # Set up summary data for detection counts.
    summary_data = {
        'unique_persons': [],
        'unique_weapons': [],
        'activity_counts': {label: 0 for label in class_labels.values()}
    }
    summary_lock = Lock()
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print("Error: Could not open the output video file for writing.")
        return

    frame_index = 0
    last_state = {'last_activity': None}  # To track activity changes between frames
    
    # Use a ThreadPoolExecutor to process frames concurrently.
    futures = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            # Submit the frame processing job; pass both locks and shared summary_data.
            futures[frame_index] = executor.submit(
                process_frame, frame, frame_index, fps, video_name,
                cursor, db_lock, summary_data, summary_lock, last_state
            )
        
        # Write processed frames in order.
        for i in range(1, frame_index + 1):
            processed_frame = futures[i].result()
            out.write(processed_frame)
    
    cap.release()
    out.release()
    conn.close()
    
    # After processing all frames, record the aggregated detection counts
    report_db_path = 'D:\\FYP\\Violent-activities-detection-and-scene-understanding\\Prediction Script\\Timestamps\\Report_Requirments.db'
    count_conn = sqlite3.connect(report_db_path, check_same_thread=False)
    count_cursor = count_conn.cursor()
    count_cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT,
            activity TEXT,
            count_person INTEGER,
            count_weapon INTEGER
        )
    ''')
    count_conn.commit()
    
    # Determine the most frequent detected activity
    most_frequent_activity = max(summary_data['activity_counts'], key=summary_data['activity_counts'].get)
    count_cursor.execute(
        'INSERT INTO detection_counts (video_name, activity, count_person, count_weapon) VALUES (?, ?, ?, ?)',
        (video_name, most_frequent_activity, len(summary_data['unique_persons']), len(summary_data['unique_weapons']))
    )
    count_conn.commit()
    count_conn.close()
    
    print("Multithreaded GPU-accelerated detection complete, output saved to:", output_video_path)

# Example usage:
if __name__ == '__main__':
    input_video = "D:\\FYP_APP\\Testing videos\\Violence.mp4"
    output_video = "D:\\FYP_APP\\Violence.webm"
    run_detection_multithread(input_video, output_video)
