import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load Model
MODEL_PATH = "model/image_classification_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Prediction Function
def predict_image(image_path):
    IMG_SIZE = (150, 150)  # Same as in training
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][class_idx]

    class_labels = {v: k for k, v in train_generator.class_indices.items()}  # Map indices to class names
    class_name = class_labels[class_idx]

    return {"class_name": class_name, "confidence": float(confidence)}
