import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Paths
LABELS_CSV = "labels.csv"   # CSV file is in the same folder as script
IMAGE_DIR = "myData"        # Folder where images are stored

# Parameters
IMG_HEIGHT, IMG_WIDTH = 32, 32
BATCH_SIZE = 32
EPOCHS = 15

# Load label names
label_df = pd.read_csv(LABELS_CSV)
num_classes = len(label_df)
print(f"Number of classes: {num_classes}")

# Load dataset
images = []
labels = []

for class_id in range(num_classes):
    class_path = os.path.join(IMAGE_DIR, str(class_id))
    if not os.path.exists(class_path):
        print(f"Warning: {class_path} does not exist.")
        continue
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        images.append(img)
        labels.append(class_id)

images = np.array(images) / 255.0
labels = to_categorical(np.array(labels), num_classes)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)

# Save model
model.save("traffic_sign_model.h5")

# Plot accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
