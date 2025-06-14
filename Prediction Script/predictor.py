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
    identify_weapon_holder, run_pose_estimation_and_save, class_labels,
    recognize_criminals_and_draw  # <-- ADDED IMPORT
)

# ==================== Alert System Configuration ====================
ALERT_API_URL = "http://127.0.0.1:3000/api/sendText"
ALERT_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}
CHAT_ID = "3113076683@c.us"

# Dictionaries to store the last time an alert was sent (for spam prevention)
last_activity_alert_sent = {}
last_criminal_alert_sent = {} # NEW: Cooldown timer for criminal alerts

def send_activity_alert(activity, camera_name):
    """Sends an alert for a violent activity."""
    current_time = datetime.now()
    if activity in last_activity_alert_sent:
        time_diff = (current_time - last_activity_alert_sent[activity]).seconds
        if time_diff < 30: # 30-second cooldown for activity alerts
            return
    data = {
        "chatId": CHAT_ID,
        "text": f"Anomaly Detected in {camera_name}: {activity}!",
        "session": "default"
    }
    try:
        requests.post(ALERT_API_URL, json=data, headers=ALERT_HEADERS)
        last_activity_alert_sent[activity] = current_time
    except requests.exceptions.RequestException as e:
        print(f"Failed to send activity alert: {e}")

def send_criminal_alert(name, cnic):
    """Sends a soft reminder alert for a detected criminal."""
    current_time = datetime.now()
    # Use CNIC as a unique key for the cooldown
    if cnic in last_criminal_alert_sent:
        time_diff = (current_time - last_criminal_alert_sent[cnic]).seconds
        if time_diff < 60: # 60-second cooldown for criminal reminders
            return
    data = {
        "chatId": CHAT_ID,
        "text": f"Reminder: A person with a potential criminal background has been detected. Name: {name} (CNIC: {cnic}). Please be careful.",
        "session": "default"
    }
    try:
        requests.post(ALERT_API_URL, json=data, headers=ALERT_HEADERS)
        last_criminal_alert_sent[cnic] = current_time
    except requests.exceptions.RequestException as e:
        print(f"Failed to send criminal alert: {e}")


# ==================== Frame Processing Function ====================
def process_frame(frame, frame_index, fps, video_name, cursor, db_lock, summary_data, summary_lock, last_state, video_id=None):
    elapsed_time = frame_index / fps
    elapsed_time_str = str(timedelta(seconds=elapsed_time))

    # Step 1: Predict activity
    detected_activity = predict_activity(frame, video_id)
    weapons, persons = detect_objects(frame)

    # Step 2: Identify weapon holder and run pose estimation
    weapon_holder = identify_weapon_holder(weapons, persons)
    weapon_holder_crop = None
    frame_height, frame_width = frame.shape[:2]
    if weapon_holder:
        px1, py1, px2, py2 = weapon_holder
        crop_img = frame[max(py1, 0):min(py2, frame_height), max(px1, 0):min(px2, frame_width)].copy()
        weapon_holder_crop = run_pose_estimation_and_save(crop_img, frame_index)

    # Step 3: Send activity alert if violent
    if detected_activity != "Normal":
        send_activity_alert(detected_activity, video_name)

    # Step 4: Update shared summary data
    with summary_lock:
        update_unique_objects(summary_data['unique_persons'], persons, threshold=50)
        update_unique_objects(summary_data['unique_weapons'], weapons, threshold=50)
        summary_data['activity_counts'][detected_activity] += 1

    # Step 5: Log to database
    if detected_activity != last_state.get('last_activity') or weapon_holder:
        with db_lock:
            cursor.execute(
                'INSERT INTO timestamps (video_name, timestamp) VALUES (?, ?)',
                (video_name, elapsed_time_str)
            )
            cursor.connection.commit()
        last_state['last_activity'] = detected_activity

    # Step 6: Run Criminal Recognition for DroidCam and send alerts
    if video_id == 2:
        detected_criminals = recognize_criminals_and_draw(frame)
        if detected_criminals:
            for criminal in detected_criminals:
                send_criminal_alert(criminal['name'], criminal['cnic'])

    # Step 7: Draw bounding boxes for persons and weapons
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for (x1, y1, x2, y2) in weapons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if weapon_holder:
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)

    # Step 8: Overlay text
    cv2.putText(frame, f"Activity: {detected_activity}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Time: {elapsed_time_str}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Overlay weapon holder crop
    if weapon_holder_crop is not None:
        overlay = cv2.resize(weapon_holder_crop, (150, 150))
        frame[0:150, frame_width-150:frame_width] = overlay

    return frame

# ==================== Multithreaded Video Processing ====================
def run_detection_multithread(video_path, output_video_path):
    """
    Processes the input video frame-by-frame using multithreading,
    leveraging GPU acceleration for inference.
    Also integrates alert notifications for detected violent activities.
    """
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

    db_lock = Lock()
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
    last_state = {'last_activity': None}

    # Default to video_id 0 for standalone script, using the first model.
    # This can be adjusted if needed.
    default_video_id_for_script = 0

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {}
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            futures[frame_index] = executor.submit(
                process_frame, frame, frame_index, fps, video_name,
                cursor, db_lock, summary_data, summary_lock, last_state,
                video_id=default_video_id_for_script
            )

        for i in range(1, frame_index + 1):
            processed_frame = futures[i].result()
            out.write(processed_frame)

    cap.release()
    out.release()
    conn.close()
    print("Detection complete:", output_video_path)

# ==================== Example Usage ====================
if __name__ == '__main__':
    input_video = "D:\\FYP_APP\\Testing videos\\Violence.mp4"
    output_video = "D:\\FYP_APP\\Violence.webm"
    run_detection_multithread(input_video, output_video)