# Violence & Criminal AI Detection System

## Overview
This repository contains the implementation of an advanced crime and violence detection system utilizing deep learning techniques. The system is designed to analyze CCTV footage, detect violent activities, recognize criminals, bookmark frames for forensic analysis, and generate AI based reports when critical incidents occur.

## Features
- **Violence Detection**: Detects and classifies various violent activities such as **abuse, arson, burglary, shooting, fighting, and vandalism**.
- **Criminal Recognition**: Identifies known criminals from the video footage using a face recognition model.
- **Frame Bookmarking**: Automatically bookmarks frames whenever a violent activity is detected, ensuring critical evidence is captured for investigation.
- **Scene Understanding & Description**: Utilizes OpenAI's CLIP model to generate detailed scene descriptions, providing contextual information about the detected activities.
- **Automated Report System**: Generates instant reports via LangChain, OpenAI & Google's FLAN-T5 for authorities.

## Model Architecture
The system is built using multiple deep learning models, including:
- **YOLO-based Object Detection Model**: Used for detecting weapons, and people in the video.
- **Vision Transformer (ViT) for Human Action Recognition**: Classifies various human actions to determine violent activities.
- **Face Recognition Model**: Identifies criminals based on a pre-existing database.
- **Scene Understanding Model**: Generates detailed descriptions of the crime scene using a state-of-the-art transformer-based architecture.

## Workflow
1. **Video Frame Processing**: The input video is divided into frames and preprocessed for analysis.
2. **Criminal Recognition**: Matches detected individuals with a database of known criminals.
3. **Violence Detection**: Each frame is analyzed to classify whether a violent activity is taking place.
4. **Weapon Detection**: Identifies the presence of weapons and other dangerous objects.
5. **Frame Bookmarking**: The system saves timestamps and frames where violent activities occur for forensic analysis.
6. **Scene Understanding**: Generates textual descriptions to explain what is happening in the scene.
7. **Alert System**: If a high-priority crime is detected, an automated alert is sent to law enforcement agencies.

## Installation
To set up the system, follow these steps:

```bash
# Clone the repository
git clone https://github.com/hisanusman/Violent-activities-detection-and-scene-understanding.git
cd Violent-activities-detection-and-scene-understanding

# Install dependencies
pip install -r requirements.txt
```

## Results & Performance
The system has been evaluated on real-world CCTV footage and achieved:
- **90%+ accuracy** in violent activity detection.
- **High precision** in weapon detection using YOLO models.
- **Robust face recognition** with a well-curated criminal database.

## Future Enhancements
- Improve scene description model using multimodal AI techniques.
- Deploy the system as a cloud-based service with real-time monitoring dashboards.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes.
4. Open a pull request.

## License
This project is licensed under the MIT License.
