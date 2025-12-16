# AkilliGoz (Smart Glasses Project) 👓🤖

This repository contains the source code for the **AkilliGoz** project, a smart glasses system powered by AI for object detection, face recognition, currency identification, and text reading. It includes code optimized for **Raspberry Pi 5 with Hailo-8L AI Kit** and a companion **Web Dashboard**.

## 📂 Project Structure

- **`main_glasses_hailo.py`**: The main application script for the Smart Glasses (Raspberry Pi 5 + Hailo AI).
  - Features: Object Detection (YOLOv8), Face Recognition (DeepFace/Facenet512), Currency Recognition, Color Detection, OCR (Text-to-Speech).
- **`web_face_recognition/`**: A Flask-based web application for managing the face database.
  - Features: User Authentication (Google OAuth), Face Registration, Dashboard, Shared Database with Glasses.
- **`datasets/`**: Directory for datasets (ignored in git).
- **`models/`**: Place to store `.pt` or `.hef` models (large models are ignored by default).

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Raspberry Pi 5 (for the glasses script) with Hailo AI Kit installed.
- Camera connected to the device.

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd akilligoz-main
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

#### 1. Web Dashboard (PC/Server)
Starts the web interface to register faces and manage users.
```bash
cd web_face_recognition
python app.py
```
Access at: `http://localhost:5000`

#### 2. Smart Glasses (Raspberry Pi)
Starts the AI assistant on the glasses.
```bash
python main_glasses_hailo.py
```
*Note: Ensure you have the necessary `.hef` models for Hailo acceleration.*

## 🛠 Technologies

- **AI/ML**: YOLOv8 (Ultralytics), DeepFace, EasyOCR, Hailo-8L.
- **Backend**: Flask, SQLAlchemy.
- **Computer Vision**: OpenCV.
- **Audio**: gTTS, Pygame.

## 📄 License

[License Information Here]
