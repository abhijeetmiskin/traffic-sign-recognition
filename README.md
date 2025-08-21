Traffic Sign Recognition








A deep learning project for Traffic Sign Recognition using TensorFlow and Keras.
This model classifies images of traffic signs and can be used for autonomous driving systems or traffic analysis.

Features

Convolutional Neural Network (CNN) for image classification

Image preprocessing & normalization

Model training, validation & saving

Prediction script for classifying custom images

Technologies Used

Python 3.10

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Pandas

Project Structure
.
├── train_traffic_signs.py   # Script for training the model
├── predict_sign.py          # Script for predicting traffic signs from images
├── dataset/                 # Folder containing training images and labels.csv
├── traffic_sign_model.keras # Saved model after training
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

Installation

Clone the repository and install dependencies:

git clone https://github.com/yourusername/traffic-sign-recognition.git
cd traffic-sign-recognition
pip install -r requirements.txt

Usage
1. Train the Model

Ensure your dataset is placed in the dataset/ folder and run:

python train_traffic_signs.py


This trains the CNN and saves the model as traffic_sign_model.keras.

2. Predict a Traffic Sign

To classify a custom image:

python predict_sign.py --image path/to/image.png

Example Output

Training Accuracy Plot

Predicted Traffic Sign Example

(Optional: Add images of training graphs and predictions here)

Requirements

Create a requirements.txt file with:

tensorflow
keras
numpy
pandas
matplotlib
opencv-python


Install dependencies:

pip install -r requirements.txt

Dataset

This project uses a dataset structured like the GTSRB (German Traffic Sign Recognition Benchmark).
It should include images and a labels.csv file mapping image IDs to class labels.