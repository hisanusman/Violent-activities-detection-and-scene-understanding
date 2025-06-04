import cv2
import os
import sqlite3
import torch
import socket
import requests
import numpy as np
from datetime import timedelta, datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from utils import (
    predict_activity, detect_objects, update_unique_objects,
    identify_weapon_holder, run_pose_estimation_and_save, class_labels
)

# ==================== Performance Configuration ====================
FRAME_SKIP = 3  # Process every nth frame
RESIZE_FACTOR = 0.5  # Reduce frame size by 50%
MAX_WORKERS = 4  # Limit thread pool size for CPU
BATCH_SIZE = 4  # Process frames in small batches
MAX_QUEUE_SIZE = BATCH_SIZE * 3  # Maximum number of frames in queue

# ==================== Alert System Configuration ====================
ALERT_ENABLED = False  # Disable alerts temporarily
ALERT_API_URL = "http://127.0.0.1:3000/api/sendText"
ALERT_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}
CHAT_ID = "3113076683@c.us"

# Dictionary to store the last time an alert was sent for each activity
last_alert_sent = {}

def send_alert(activity, camera_name):
    """Sends an alert notification via the API if a violent activity is detected."""
    if not ALERT_ENABLED:
        return
        
    try:
        current_time = datetime.now()
        if activity in last_alert_sent:
            time_diff = (current_time - last_alert_sent[activity]).seconds
            if time_diff < 30:  # Prevent duplicate alerts within 30 seconds
                return

        data = {
            "chatId": CHAT_ID,
            "text": f"Anomaly Detected in {camera_name}: {activity}!",
            "session": "default"
        }

        response = requests.post(ALERT_API_URL, json=data, headers=ALERT_HEADERS)
        last_alert_sent[activity] = current_time
    except Exception as e:
        print(f"Warning: Alert system disabled or error: {e}")

def process_frame(frame, frame_index, fps, video_name, cursor, db_lock, summary_data, summary_lock, last_state):
    """
    Processes a single frame with optimizations for CPU:
      - Resizes frame for faster processing
      - Processes only key frames
      - Optimized detection thresholds
    """
    try:
        # Skip frames based on FRAME_SKIP
        if frame_index % FRAME_SKIP != 0:
            return frame

        elapsed_time = frame_index / fps
        elapsed_time_str = str(timedelta(seconds=elapsed_time))

        # Resize frame for faster processing
        height, width = frame.shape[:2]
        new_height = int(height * RESIZE_FACTOR)
        new_width = int(width * RESIZE_FACTOR)
        small_frame = cv2.resize(frame, (new_width, new_height))

        # Step 1: Predict activity (hardcode as "normal" for droidcam)
        if video_name == "droidcam_live":
            detected_activity = "normal"
            # Reset activity counts for droidcam to ensure only "normal" is counted
            with summary_lock:
                summary_data['activity_counts'] = {label: 0 for label in class_labels.values()}
                summary_data['activity_counts']["normal"] = frame_index // FRAME_SKIP  # Approximate count based on processed frames
        else:
            detected_activity = predict_activity(small_frame)
        
        # Only run object detection if activity is not normal and not droidcam
        weapons, persons = [], []
        if detected_activity != "normal" and video_name != "droidcam_live":
            weapons, persons = detect_objects(small_frame)
            # Scale bounding boxes back to original size
            weapons = [(int(x1/RESIZE_FACTOR), int(y1/RESIZE_FACTOR), 
                       int(x2/RESIZE_FACTOR), int(y2/RESIZE_FACTOR)) for x1, y1, x2, y2 in weapons]
            persons = [(int(x1/RESIZE_FACTOR), int(y1/RESIZE_FACTOR), 
                       int(x2/RESIZE_FACTOR), int(y2/RESIZE_FACTOR)) for x1, y1, x2, y2 in persons]

        # Step 2: Identify weapon holder (only if weapons detected and not droidcam)
        weapon_holder = None
        weapon_holder_crop = None
        if weapons and persons and video_name != "droidcam_live":
            weapon_holder = identify_weapon_holder(tuple(weapons), tuple(persons))
            if weapon_holder:
                px1, py1, px2, py2 = weapon_holder
                crop_img = frame[max(py1, 0):min(py2, frame.shape[0]), 
                               max(px1, 0):min(px2, frame.shape[1])].copy()
                weapon_holder_crop = run_pose_estimation_and_save(crop_img, frame_index)
                # Store the frame index when weapon holder was detected
                last_state['last_weapon_holder_frame'] = frame_index
                last_state['last_weapon_holder_crop'] = weapon_holder_crop
            elif 'last_weapon_holder_frame' in last_state:
                # If within 5 frames of last detection, keep showing the last crop
                if frame_index - last_state['last_weapon_holder_frame'] <= 5:
                    weapon_holder_crop = last_state['last_weapon_holder_crop']
                else:
                    # Clear the weapon holder data after 5 frames
                    last_state.pop('last_weapon_holder_frame', None)
                    last_state.pop('last_weapon_holder_crop', None)

        # Step 3: Send alert if activity is violent (skip for droidcam)
        if detected_activity != "normal" and video_name != "droidcam_live":
            send_alert(detected_activity, video_name)

        # Step 4: Update shared summary data (only if detections exist and not droidcam)
        if video_name != "droidcam_live":
            with summary_lock:
                if persons:
                    update_unique_objects(summary_data['unique_persons'], persons, threshold=50)
                if weapons:
                    update_unique_objects(summary_data['unique_weapons'], weapons, threshold=50)
                summary_data['activity_counts'][detected_activity] += 1

        # Step 5: Log to database if activity changes or a weapon is detected (skip weapon detection for droidcam)
        if detected_activity != last_state.get('last_activity') or (weapon_holder and video_name != "droidcam_live"):
            with db_lock:
                cursor.execute(
                    'INSERT INTO timestamps (video_name, timestamp) VALUES (?, ?)',
                    (video_name, elapsed_time_str)
                )
                cursor.connection.commit()
            last_state['last_activity'] = detected_activity

        # Step 6: Draw bounding boxes (only if detections exist and not droidcam)
        if video_name != "droidcam_live":
            if persons:
                for (x1, y1, x2, y2) in persons:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if weapons:
                for (x1, y1, x2, y2) in weapons:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if weapon_holder:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)

        # Step 7: Overlay text
        cv2.putText(frame, f"Activity: {detected_activity}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Overlay weapon holder crop (if exists and not droidcam)
        if weapon_holder_crop is not None and video_name != "droidcam_live":
            overlay = cv2.resize(weapon_holder_crop, (150, 150))
            frame[0:150, frame.shape[1]-150:frame.shape[1]] = overlay

    except Exception as e:
        print(f"Error processing frame {frame_index}: {e}")
        cv2.putText(frame, "Error processing frame", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

def run_detection_multithread(video_path, output_video_path):
    """
    Processes the input video with CPU optimizations:
      - Reduced thread count
      - Frame skipping
      - Optimized I/O operations
    """
    # Use relative path for database
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ts_db_path = os.path.join(current_dir, 'Timestamps', 'detection_timestamps.db')
    os.makedirs(os.path.dirname(ts_db_path), exist_ok=True)
    
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

    db_lock = Lock()
    summary_data = {
        'unique_persons': [],
        'unique_weapons': [],
        'activity_counts': {label: 0 for label in class_labels.values()}
    }
    summary_lock = Lock()

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    
    # Set OpenCV buffer size
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Adjust output FPS based on frame skip
    output_fps = fps // FRAME_SKIP
    
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (frame_width, frame_height))

    if not out.isOpened():
        print("Error: Could not open the output video file for writing.")
        return

    frame_index = 0
    last_state = {'last_activity': None}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        while cap.isOpened():
            frames_batch = []
            for _ in range(BATCH_SIZE):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_index += 1
                frames_batch.append((frame, frame_index))

            if not frames_batch:
                break

            # Submit batch for processing
            for frame, idx in frames_batch:
                futures[idx] = executor.submit(
                    process_frame, frame, idx, fps, video_name,
                    cursor, db_lock, summary_data, summary_lock, last_state
                )

            # Write processed frames in order
            while len(futures) > MAX_QUEUE_SIZE:
                next_frame_idx = min(futures.keys())
                processed_frame = futures.pop(next_frame_idx).result()
                if next_frame_idx % FRAME_SKIP == 0:
                    out.write(processed_frame)

        # Process remaining frames
        for idx in sorted(futures.keys()):
            processed_frame = futures[idx].result()
            if idx % FRAME_SKIP == 0:
                out.write(processed_frame)

    cap.release()
    out.release()
    conn.close()
    print("Detection complete:", output_video_path)

if __name__ == '__main__':
    input_video = "D:\\FYP_APP\\Testing videos\\Violence.mp4"
    output_video = "D:\\FYP_APP\\Violence.webm"
    run_detection_multithread(input_video, output_video)
