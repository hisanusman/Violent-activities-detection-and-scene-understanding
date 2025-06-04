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
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel, AutoModelForSeq2SeqLM, AutoTokenizer, ViTForImageClassification
from datetime import datetime, timedelta
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import functools
import hashlib
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ==================== Performance Configuration ====================
# Use model half precision for faster CPU inference
HALF_PRECISION = True
# Cache size for function results
CACHE_SIZE = 100
# Minimum confidence thresholds
MIN_DETECTION_CONFIDENCE = 0.70
MIN_POSE_CONFIDENCE = 0.5

# ==================== Model Initialization ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llm = OpenAI(api_key="API")  # Replace with your OpenAI API key

# Initialize FLAN-T5 model and tokenizer (fallback)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize CLIP model for scene understanding
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# Scene categories for CLIP
scene_categories = [
    "people fighting violently with each other",
    "physical abuse scene with a person hurting another person",
    "arson or fire setting with flames and smoke",
    "burglary in progress with a person breaking into a building",
    "shooting with firearms",
    "vandalism of property with a person damaging property",
    "normal peaceful scene"
]

# Define scene mapping
clip_class_mapping = {
    "people fighting violently with each other": "fighting",
    "physical abuse scene with a person hurting another person": "abuse",
    "arson or fire setting with flames and smoke": "arson",
    "burglary in progress with a person breaking into a building": "burglary",
    "shooting with firearms": "shooting",
    "vandalism of property with a person damaging property": "vandalism",
    "normal peaceful scene": "normal"
}

# Initialize FLAN-T5 model and tokenizer (fallback)
flan_model_name = "google/flan-t5-base"
flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_name)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_name).to(device)

if HALF_PRECISION and device == "cpu":
    flan_model = flan_model.half()

# ==================== Utility Functions ====================
def predict_activity_with_clip(frame):
    """Predicts the activity in a frame using the CLIP model."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    inputs = clip_processor(
        text=scene_categories,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        top_prob, top_idx = torch.max(probs, dim=1)
        scene_category = scene_categories[top_idx.item()]
        confidence = top_prob.item()
        
    activity = clip_class_mapping.get(scene_category, "normal")
    return activity, confidence

def predict_activity(frame):
    """Predicts the activity label for a given frame using CLIP model."""
    return predict_activity_with_clip(frame)[0]

# Define Image Transformations for ViT with caching
@functools.lru_cache(maxsize=CACHE_SIZE)
def transform_image(image_array):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)))

# Class Labels for ViT Model
class_labels = {
    0: 'fighting',
    1: 'abuse',
    2: 'arson',
    3: 'burglary',
    4: 'shooting',
    5: 'vandalism',
    6: 'normal'
}

# Load ViT Model for Activity Recognition
current_dir = os.path.dirname(os.path.abspath(__file__))
vit_model_path = os.path.join(current_dir, "vit_anomaly_detector (1).pth")
vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", num_labels=6, ignore_mismatched_sizes=True
)

try:
    vit_model.load_state_dict(torch.load(vit_model_path, map_location=device))
    print(f"Successfully loaded ViT model from {vit_model_path}")
    if HALF_PRECISION and device == "cpu":
        vit_model = vit_model.half()
except FileNotFoundError:
    print(f"Warning: ViT model file not found at {vit_model_path}. Using default weights.")
except Exception as e:
    print(f"Warning: Error loading ViT model: {e}. Using default weights.")

vit_model.to(device).eval()

# Load Weapon (and Person) Detection Model (YOLOv8)
weapon_model_path = os.path.join(current_dir, "best_weapons.pt")
try:
    weapon_model = YOLO(weapon_model_path)
    weapon_model.to(device)
    if HALF_PRECISION and device == "cpu":
        weapon_model = weapon_model.half()
    print(f"Successfully loaded weapon detection model from {weapon_model_path}")
except Exception as e:
    print(f"Warning: Error loading weapon detection model: {e}")
    weapon_model = None

# Initialize MediaPipe Pose with optimized settings
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=0,  # Use the fastest model
    min_detection_confidence=MIN_POSE_CONFIDENCE
)

# Create output folder for saved weapon holder images
output_img_folder = os.path.join(current_dir, "weapon_holder_images")
os.makedirs(output_img_folder, exist_ok=True)

# Load Criminal Embeddings with caching
@functools.lru_cache(maxsize=1)
def load_criminal_embeddings():
    embeddings_path = os.path.join(current_dir, "embeddings.pkl")
    try:
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
        print(f"Successfully loaded criminal embeddings from {embeddings_path}")
        return embeddings
    except Exception as e:
        print(f"Warning: Error loading criminal embeddings: {e}. Using empty list.")
        return []

criminal_embeddings = load_criminal_embeddings()

# Load YOLO face detection model
face_model_path = os.path.join(current_dir, "yolov8n-face.pt")
try:
    face_model = YOLO(face_model_path)
    if HALF_PRECISION and device == "cpu":
        face_model = face_model.half()
    print(f"Successfully loaded face detection model from {face_model_path}")
except Exception as e:
    print(f"Warning: Error loading face detection model: {e}")
    face_model = None

def frame_to_hash(frame):
    """Convert frame to a hashable representation."""
    return hashlib.md5(frame.tobytes()).hexdigest()

# Define Image Transformations for ViT with caching
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@functools.lru_cache(maxsize=CACHE_SIZE)
def transform_and_predict(frame_hash):
    """Transform and predict activity for a frame using its hash."""
    frame = cache_frames.get(frame_hash)
    if frame is None:
        return "normal"
    
    try:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0).to(device)
        if HALF_PRECISION and device == "cpu":
            image = image.half()
        
        with torch.no_grad():
            output = vit_model(image)
            logits = output.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        return class_labels.get(predicted_class, "normal")
    except Exception as e:
        print(f"Warning: Error predicting activity: {e}")
        return "normal"

# Initialize cache for frames
cache_frames = {}

def predict_activity(frame):
    """Predicts the activity label for a given frame using the ViT model with caching."""
    frame_hash = frame_to_hash(frame)
    cache_frames[frame_hash] = frame
    result = transform_and_predict(frame_hash)
    # Clean up cache
    if len(cache_frames) > CACHE_SIZE:
        old_hash = next(iter(cache_frames))
        del cache_frames[old_hash]
    return result

@functools.lru_cache(maxsize=CACHE_SIZE)
def detect_objects_cached(frame_hash):
    """Cached version of object detection."""
    frame = cache_frames.get(frame_hash)
    if frame is None:
        return [], []
    
    weapons, persons = [], []
    try:
        if weapon_model is not None:
            results = weapon_model(frame, conf=MIN_DETECTION_CONFIDENCE)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    if cls == 1 and conf >= MIN_DETECTION_CONFIDENCE:  # Weapon
                        weapons.append((x1, y1, x2, y2))
                    elif cls == 0 and conf >= MIN_DETECTION_CONFIDENCE:  # Person
                        persons.append((x1, y1, x2, y2))
    except Exception as e:
        print(f"Warning: Error detecting objects: {e}")
    return tuple(weapons), tuple(persons)

def detect_objects(frame):
    """Wrapper for cached object detection."""
    frame_hash = frame_to_hash(frame)
    cache_frames[frame_hash] = frame
    weapons, persons = detect_objects_cached(frame_hash)
    # Clean up cache
    if len(cache_frames) > CACHE_SIZE:
        old_hash = next(iter(cache_frames))
        del cache_frames[old_hash]
    return list(weapons), list(persons)

@functools.lru_cache(maxsize=CACHE_SIZE)
def identify_weapon_holder(weapons_tuple, persons_tuple):
    """Returns the bounding box of a person holding a weapon if found (with caching)."""
    for wx1, wy1, wx2, wy2 in weapons_tuple:
        for px1, py1, px2, py2 in persons_tuple:
            if px1 < wx1 < px2 and py1 < wy1 < py2:
                return (px1, py1, px2, py2)
    return None

def update_unique_objects(unique_list, detections, threshold=50):
    """Updates the list of unique objects with distance-based deduplication."""
    if not detections:
        return
    
    for box in detections:
        x1, y1, x2, y2 = box
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Use numpy for faster distance calculation
        if unique_list:
            centers = np.array(unique_list)
            distances = np.sqrt(np.sum((centers - np.array(center)) ** 2, axis=1))
            if np.min(distances) >= threshold:
                unique_list.append(center)
        else:
            unique_list.append(center)

def run_pose_estimation_and_save(crop_img, frame_index):
    """Runs MediaPipe Pose estimation with optimized settings."""
    try:
        # Resize image for faster processing if too large
        max_size = 400
        h, w = crop_img.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            crop_img = cv2.resize(crop_img, (int(w * scale), int(h * scale)))
        
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(crop_rgb)
        
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                crop_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        output_path = os.path.join(output_img_folder, f"weapon_holder_frame_{frame_index}.jpg")
        cv2.imwrite(output_path, crop_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return crop_img
    except Exception as e:
        print(f"Warning: Error in pose estimation: {e}")
        return crop_img

# ==================== Report Generation Functions ====================

def generate_scene_description_with_openai(activity):
    """Generates a scene description using OpenAI's GPT-4 API via LangChain."""
    prompt_template = PromptTemplate(
        input_variables=["activity"],
        template=(
            "You are an expert crime scene investigator analyzing violent CCTV footage. "
            "The detected activity is '{activity}'. "
            "Write a detailed and professional crime scene report of at least 6 to 7 lines (at least), describing the event in a formal manner. "
            "Include relevant details such as the nature of the activity and details that could help crime investigations."
        ),
    )
    prompt = prompt_template.format(activity=activity)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert crime scene investigator analyzing a CCTV footage."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def generate_scene_description_with_flan(activity):
    """Generates a scene description using FLAN-T5."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    
    # Load FLAN-T5 model and tokenizer if not already loaded
    flan_model_name = "google/flan-t5-base"
    flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_name)
    flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_name).to(device)
    
    prompt = (
        f"You are an expert crime scene investigator analyzing violent CCTV footage. "
        f"The detected activity is '{activity}'. "
        f"Write a detailed and professional crime scene report of at least 6 to 7 lines (at least), describing the event in a formal manner. "
        f"Ensure that the report is useful for a police investigation."
        f"Include relevant details such as the nature of the activity and details that could help crime investigations."
    )
    inputs = flan_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = flan_model.generate(**inputs, max_length=1500)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_scene_description(activity):
    """Generates a scene description using FLAN-T5 and templates."""
    try:
        ai_description = generate_scene_description_with_openai(activity)
    except Exception as e:
        print(f"OpenAI API error: {e}. Falling back to FLAN-T5.")
        ai_description = generate_scene_description_with_flan(activity)
    
    intro_templates = {
        "fighting": "Analysis of CCTV footage has documented a physical altercation classified as {activity}.",
        "abuse": "CCTV footage analysis reveals evidence of abusive behavior classified as {activity}.",
        "arson": "CCTV footage captured evidence of intentional fire-setting classified as {activity}.",
        "burglary": "Security camera footage has documented a breaking and entering incident classified as {activity}.",
        "shooting": "CCTV analysis has documented a firearms discharge incident classified as {activity}.",
        "vandalism": "Video evidence shows property damage incident classified as {activity}.",
        "normal": "CCTV footage analysis has documented everything as {activity}."
    }
    
    severity_templates = {
        "fighting": "This incident is classified as a physical assault case.",
        "abuse": "This incident is classified as an abuse case.",
        "arson": "This incident is classified as arson, a serious felony offense.",
        "burglary": "This incident is classified as burglary, a felony offense.",
        "shooting": "This incident is classified as a firearms offense.",
        "vandalism": "This incident is classified as vandalism or criminal damage to property.",
        "normal": "This incident suggests that everything is normal and smooth."
    }
    
    recommendation_template = """Further investigation is recommended, including:
    1. Collection of additional footage from nearby cameras to track participant movements
    2. Interviews with any witnesses present during the incident
    3. Correlation with any reported incidents in the area during the same timeframe."""
    
    # Generate additional details using the templates
    intro = intro_templates.get(activity, intro_templates["normal"]).format(activity=activity)
    severity = severity_templates.get(activity, severity_templates["normal"])
    
    template_details = f"{intro}\n\n{severity}\n\n{recommendation_template}"
    
    # Combine the AI-generated description with the additional template details
    full_description = f"{ai_description}\n\nAdditional Details:\n{template_details}"
    return full_description

def generate_pdf_report(activity, scene_description, output_path):
    """Generates a formal crime scene report PDF."""
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

    # Date and Location
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "Date:")
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, date_str, ln=True)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(40, 10, "Location:")
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "Not Specified", ln=True)

    pdf.ln(5)

    # Crime Details
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "Detected Crime:")
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, activity, ln=True)

    pdf.ln(10)

    # Scene Description
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Scene Description:", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, scene_description)

    pdf.ln(10)
    # Footer
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Report generated on {date_str}", align="C")

    pdf.output(output_path)