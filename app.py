import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import os
from torchvision import models

# Initialize Mediapipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Load PyTorch emotion recognition model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model = models.resnet18(pretrained=False)
emotion_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = emotion_model.fc.in_features
emotion_model.fc = nn.Linear(num_ftrs, 7)
try:
    emotion_model.load_state_dict(torch.load('emotion_model.pth', map_location=device, weights_only=True))
except FileNotFoundError:
    st.error("Model file 'emotion_model.pth' not found in D:\Computer_Vision\. Please ensure it exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
emotion_model.to(device)
emotion_model.eval()
emotion_labels = ['Happy', 'Sad', 'Neutral', 'Surprise', 'Angry', 'Disgust', 'Fear']

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load background images
backgrounds = {
    'Happy': 'backgrounds/happy.jpg',
    'Sad': 'backgrounds/sad.jpeg',
    'Neutral': 'backgrounds/neutral.jpeg',
    'Surprise': 'backgrounds/surprise.jpg',
    'Angry': 'backgrounds/angry.jpeg',
    'Disgust': 'backgrounds/disgust.jpg',
    'Fear': 'backgrounds/fear.jpeg'
}
loaded_backgrounds = {}

# Initialize webcam to get resolution
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Cannot access webcam. Please check your camera connection.")
    st.stop()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Load and resize background images
for key, path in backgrounds.items():
    full_path = os.path.join('D:/Computer_Vision', path)
    if os.path.exists(full_path):
        img = cv2.imread(full_path)
        if img is not None:
            loaded_backgrounds[key] = cv2.resize(img, (frame_width, frame_height))
        else:
            st.warning(f"Failed to load background for {key}. Using default.")
            loaded_backgrounds[key] = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    else:
        st.warning(f"Background file {full_path} not found. Using default.")
        loaded_backgrounds[key] = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

def process_frame(frame):
    # Convert frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform selfie segmentation
    results = selfie_segmentation.process(frame_rgb)
    mask = results.segmentation_mask > 0.9  # Threshold for foreground

    # Extract face for emotion recognition
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        st.error("Failed to load Haar cascade file.")
        return frame, "Error"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion = 'Neutral'  # Default emotion
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]

        # Preprocess face for emotion model
        face_pil = Image.fromarray(face)
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # Predict emotion
        with torch.no_grad():
            outputs = emotion_model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_labels[predicted.item()]

    # Apply background based on emotion
    background = loaded_backgrounds.get(emotion, loaded_backgrounds['Neutral'])
    output_frame = np.where(mask[:, :, None], frame, background)

    return output_frame, emotion

# Streamlit app
st.title("Emotion-Aware Virtual Background Generator")
st.write("This app detects your facial emotions and changes the background accordingly.")
run = st.checkbox("Run Webcam")

# Create two columns for side-by-side display
col1, col2 = st.columns(2)
with col1:
    st.write("Original Feed")
    original_placeholder = st.empty()
with col2:
    st.write("Processed Feed")
    processed_placeholder = st.empty()
emotion_placeholder = st.empty()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Cannot access webcam. Please check your camera connection.")
    st.stop()

# Main loop
while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame.")
        break

    # Process frame
    output_frame, emotion = process_frame(frame)

    # Convert frames to RGB for Streamlit
    original_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

    # Display frames
    original_placeholder.image(original_frame_rgb, channels="RGB")
    processed_placeholder.image(processed_frame_rgb, channels="RGB")
    emotion_placeholder.write(f"Detected Emotion: {emotion}")

# Release resources
cap.release()