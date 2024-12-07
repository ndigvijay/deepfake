# features.py
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


model_path = '/Users/digvijaynarayan/Desktop/personal/vada/deepfakedemo/models/deepfake.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")


model = load_model(model_path)

print(model.input_shape)


def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image.")
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detect face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
  
    face_size = (128, 128) 
    face = cv2.resize(face, face_size)
    # Normalize the image
    face = face.astype('float32') / 255.0
    # Expand dimensions to match the input shape (1, height, width, channels)
    face = np.expand_dims(face, axis=0)
    return face

def image_classifier(image_path):
    try:
        # Preprocess the image
        input_face = preprocess_image(image_path)
        # Make prediction
        prediction = model.predict(input_face)
        fake_probability = prediction[0][0]
        threshold = 0.5 
        if fake_probability >= threshold:
            result = 1  
        else:
            result = 0  
        return result
    except Exception as e:
        print(f"Error in image_classifier: {e}")
        raise e

