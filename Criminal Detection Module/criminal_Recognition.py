import os
import cv2
import numpy as np
import pandas as pd
import pickle
from deepface import DeepFace
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

def get_face_embeddings(image_folder, metadata_file):
    embeddings = []
    metadata = pd.read_csv(metadata_file)

    for idx, row in metadata.iterrows():
        folder_path = os.path.join(image_folder, row["folder"])
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            try:
                embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)
                if isinstance(embedding, list):
                    # Append embedding with metadata
                    embeddings.append((embedding[0]["embedding"], row["name"], row["cnic"], row["age"]))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    return embeddings

def is_criminal(new_embedding, criminal_embeddings, threshold=0.6):
    # Reshape new_embedding to 2D
    new_embedding = np.array(new_embedding).reshape(1, -1)

    similarities = []
    for e in criminal_embeddings:
        emb = np.array(e[0]).reshape(1, -1)  # Ensure criminal embeddings are 2D
        similarities.append(cosine_similarity(new_embedding, emb)[0, 0])

    max_similarity = max(similarities)

    if max_similarity > threshold:
        index = np.argmax(similarities)
        _, name, cnic, age = criminal_embeddings[index]
        return True, max_similarity, name, cnic, age

    return False, max_similarity, None, None, None

def main(video_path,output_path, model):
    with open("D:\\Datasets\\Facial Recognition Augmented\\embeddings.pkl", "rb") as f:
        criminal_embeddings = pickle.load(f)
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create a VideoWriter object
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)
        detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else [1.0] * len(detections)

        for detection, confidence in zip(detections, confidences):
            x1, y1, x2, y2 = map(int, detection[:4])  # Bounding box coordinates
            if confidence < 0.57:  # Confidence threshold
                continue

            # Extract face ROI
            face_img = frame[y1:y2, x1:x2]

            # Get embedding for the detected face
            try:
                new_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)
                if isinstance(new_embedding, list):
                    new_embedding = new_embedding[0]["embedding"]
            except Exception as e:
                print(f"Error processing face: {e}")
                continue

            if new_embedding is not None:
                # Check if the face matches a criminal embedding
                criminal_detected, similarity, name, cnic, age = is_criminal(new_embedding, criminal_embeddings, threshold=0.58)

                if criminal_detected:
                    # Draw bounding box and metadata
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box
                    cv2.putText(frame, f"{name}, {age} yrs", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"CNIC: {cnic}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Write the processed frame to the output video
        out.write(frame)

        # Display the frame (optional for debugging)
        #cv2.imshow("Video", frame)
        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break

    # Release resources
    cap.release()
    out.release()
    #cv2.destroyAllWindows()

    print(f"Processed video saved to {output_path}")


if __name__ == "__main__":
    model = YOLO("D:\\Datasets\\Facial Recognition Augmented\\yolov8n-face.pt")  # Replace with your YOLO model file path
    # Open the video file
    video_path = "D:\\Datasets\\test\\test_video_2.mp4"
    output_path = "D:\\Datasets\\Outputs\\output_video_2.mp4"
    main(video_path,output_path, model)