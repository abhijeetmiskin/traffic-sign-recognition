import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys

# ----- CONFIG -----
MODEL_PATH = "traffic_sign_model.h5"   # or traffic_sign_model.keras
IMG_PATH = "9.jpeg"            # Replace with your test image path
IMG_SIZE = (32, 32)                    # Must match training size
LABELS_CSV = "labels.csv"              # CSV mapping class IDs to names

# Load model
model = load_model(MODEL_PATH)

# Load labels
import pandas as pd
labels_df = pd.read_csv(LABELS_CSV)
id_to_label = dict(zip(labels_df['ClassId'], labels_df['Name']))

# Load and preprocess image
img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize same as training

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)

print(f"Predicted Class ID: {predicted_class}")
print(f"Class Name: {id_to_label[predicted_class]}")
print(f"Confidence: {confidence:.2f}")
