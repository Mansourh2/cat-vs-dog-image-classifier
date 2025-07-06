import tensorflow as tf
import numpy as np
from PIL import Image

# Load model and handle errors
print("Loading model...")
try:
    model = tf.keras.models.load_model("keras_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load labels and handle errors
print("Loading labels...")
try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print("Labels loaded successfully.")
except Exception as e:
    print(f"Error loading labels file: {e}")
    exit()

# Preprocess image function
def preprocess_image(image_path):
    print(f"Processing image: {image_path}")
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        print("Image processed successfully.")
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error processing image: {e}")
        exit()

# Image path
image_path = "test_image.jpg"
input_image = preprocess_image(image_path)

# Make predictions
print("Making predictions...")
predictions = model.predict(input_image)
predicted_class = class_names[np.argmax(predictions)]

# Print predicted class
print(f"Predicted class: {predicted_class}")