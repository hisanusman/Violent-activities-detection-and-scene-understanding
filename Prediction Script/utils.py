import os
import cv2
import math
import torch
import openai
import numpy as np
from PIL import Image
from fpdf import FPDF
import mediapipe as mp
from ultralytics import YOLO
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import torchvision.transforms as transforms
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, ViTForImageClassification, CLIPProcessor, CLIPModel
from datetime import datetime, timedelta
import pickle
from sklearn.metrics.pairwise import cosine_similarity
# ==================== Utility Functions ====================
# Initialize OpenAI / LangChain LLM (for report generation)
llm = OpenAI(api_key="API_KEY")
# Initialize FLAN-T5 model and tokenizer (fallback)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flan_model_name = "google/flan-t5-base"
flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_name)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_name).to(device)

# Define Image Transformations for ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Class Labels for ViT Model
class_labels = {
    0: 'fighting',
    1: 'abuse',
    2: 'arson',
    3: 'burglary',
    4: 'shooting',
    5: 'vandalism'
}

# Load ViT Model for Activity Recognition
vit_model_path = "vit_anomaly_detector (1).pth" 
vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", num_labels=6, ignore_mismatched_sizes=True
)

vit_model.load_state_dict(torch.load(vit_model_path, map_location=device))
vit_model.to(device).eval()

# Load Weapon (and Person) Detection Model (YOLOv8)
weapon_model = YOLO("best_weapons.pt")
weapon_model.to(device)
# Disable fusing to prevent removal of non-existent batchnorm layers.
try:
    weapon_model.model.fuse = lambda verbose=True: weapon_model.model
    print("Model fuse disabled successfully.")
except Exception as e:
    print("Failed to override fuse:", e)

# Initialize MediaPipe Pose for pose estimation
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Create output folder for saved weapon holder images
output_img_folder = "weapon_holder_images"
os.makedirs(output_img_folder, exist_ok=True)


# ==================== Load Criminal Embeddings ====================
with open("D:\\Datasets\\Facial Recognition Augmented\\embeddings.pkl", "rb") as f:
    criminal_embeddings = pickle.load(f)

# Load YOLO face detection model
face_model = YOLO("D:\\Datasets\\Facial Recognition Augmented\\yolov8n-face.pt")

def is_criminal(new_embedding, threshold=0.58):
    """Checks if a detected face matches a known criminal."""
    new_embedding = np.array(new_embedding).reshape(1, -1)
    similarities = []

    for e in criminal_embeddings:
        emb = np.array(e[0]).reshape(1, -1)
        similarities.append(cosine_similarity(new_embedding, emb)[0, 0])

    max_similarity = max(similarities)

    if max_similarity > threshold:
        index = np.argmax(similarities)
        _, name, cnic, age = criminal_embeddings[index]
        return True, max_similarity, name, cnic, age

    return False, max_similarity, None, None, None

def predict_activity(frame):
    """Predicts the activity label for a given frame using the ViT model."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = vit_model(image)
        logits = output.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return class_labels.get(predicted_class, "normal")

def detect_objects(frame):
    """
    Runs the YOLO model on the given frame and returns two lists:
      - weapons: list of bounding boxes for weapons
      - persons: list of bounding boxes for persons
    Bounding boxes are tuples of (x1, y1, x2, y2).
    """
    results = weapon_model(frame)
    weapons, persons = [], []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            if cls == 1 and conf >= 0.75:  # Assuming Class 1 is 'Weapon'
                weapons.append((x1, y1, x2, y2))
            elif cls == 0 and conf >= 0.8:  # Assuming Class 0 is 'Person'
                persons.append((x1, y1, x2, y2))
    return weapons, persons

def identify_weapon_holder(weapons, persons):
    """Returns the bounding box of a person holding a weapon if found."""
    for wx1, wy1, wx2, wy2 in weapons:
        for px1, py1, px2, py2 in persons:
            # If the weapon's top-left is within the person's bounding box
            if px1 < wx1 < px2 and py1 < wy1 < py2:
                return (px1, py1, px2, py2)
    return None

def update_unique_objects(unique_list, detections, threshold=50):
    """
    Updates the list of unique objects based on detection bounding boxes.
    For each detection, the centroid is calculated and compared with centroids in unique_list.
    If the distance is greater than 'threshold' from all existing ones, it is added.
    """
    for box in detections:
        x1, y1, x2, y2 = box
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        found = False
        for u_center in unique_list:
            distance = math.sqrt((center[0] - u_center[0]) ** 2 + (center[1] - u_center[1]) ** 2)
            if distance < threshold:
                found = True
                break
        if not found:
            unique_list.append(center)

def run_pose_estimation_and_save(crop_img, frame_index):
    """
    Runs MediaPipe Pose estimation on the given image crop,
    draws pose landmarks on it, and saves the image.
    Returns the annotated image.
    """
    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(crop_rgb)
    annotated_image = crop_img.copy()
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    output_path = os.path.join(output_img_folder, f"weapon_holder_frame_{frame_index}.jpg")
    cv2.imwrite(output_path, annotated_image)
    return annotated_image

# ==================== Report Generation Functions ====================

def generate_scene_description_with_openai(activity, num_people, num_weapons):
    """Generates a scene description using OpenAI's API via LangChain."""
    prompt_template = PromptTemplate(
        input_variables=["activity", "num_people", "num_weapons"],
        template=(
            "You are a crime scene investigator analyzing a video. "
            "The detected activity is '{activity}'. "
            "There are {num_people} people involved and {num_weapons} weapons detected. "
            "Write a detailed and professional crime scene report describing the event in a formal manner."
        ),
    )
    prompt = prompt_template.format(activity=activity, num_people=num_people, num_weapons=num_weapons)
    response = llm.predict(prompt)
    return response

def generate_scene_description_with_flan(activity, num_people, num_weapons):
    """Generates a scene description using FLAN-T5."""
    prompt = (
        f"You are a crime scene investigator analyzing a video. "
        f"The detected activity is '{activity}'. "
        f"There are {num_people} people involved and {num_weapons} weapons detected. "
        f"Write a detailed and professional crime scene report of at least 4 to 6 lines, describing the event in a formal manner so that it may be helpful in a police investigation."
    )
    inputs = flan_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = flan_model.generate(**inputs, max_length=1000)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_scene_description(activity, num_people, num_weapons):
    """Generates a scene description using OpenAI's API, with FLAN-T5 as a fallback."""
    try:
        return generate_scene_description_with_openai(activity, num_people, num_weapons)
    except Exception as e:
        print(f"OpenAI API error: {e}. Falling back to FLAN-T5.")
        return generate_scene_description_with_flan(activity, num_people, num_weapons)

def generate_pdf_report(activity, scene_description, num_people, num_weapons, output_path):
    """
    Generates a formal crime scene report PDF in a custom format.
    The layout includes:
      - Title ("Crime Scene Report")
      - Date, Time, and Location fields
      - Detected Crime, People Involved, Weapons Found
      - A multi-line scene description
    """
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()

    # Report Title
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, "Crime Scene Report", ln=True, align="C")
    pdf.ln(5)

    # Horizontal line
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)

    # Date, Time, and Location
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "Date:")
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, date_str, ln=True)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "Time:")
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, time_str, ln=True)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "Location:")
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "Not Specified", ln=True)  # Placeholder

    pdf.ln(5)

    # Crime Details
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "Detected Crime:")
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, activity, ln=True)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "People Involved:")
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, str(num_people), ln=True)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "Weapons Found:")
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, str(num_weapons), ln=True)

    pdf.ln(10)

    # Scene Description Header
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Scene Description:", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, scene_description)

    pdf.ln(10)
    # Footer with generation timestamp
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Report generated on {date_str} at {time_str}", align="C")

    pdf.output(output_path)