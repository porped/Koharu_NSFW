import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / r'models' / r'TF2_human_anime_model.h5'

# Load the trained model
model = load_model(model_path)  # Replace with the path to your trained model

# Specify the path to the folder containing the test images
image_folder_path = BASE_DIR / r'test_image'

# Specify the path to the folder where you want to save the classified images
output_folder_path = BASE_DIR / r'output' / r'classified_anime-human'

class_labels = {0: 'anime', 1: 'human'}

# Create the output folders for each class if they don't exist
for class_name in class_labels.values():
    class_folder = output_folder_path / class_name
    os.makedirs(class_folder, exist_ok=True)

# Iterate through the images in the test folder
for filename in os.listdir(image_folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png', 'jfif')):  # Filter for image files
        img_path = os.path.join(image_folder_path, filename)
        
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize the image

        # Make predictions
        predictions = model.predict(img)

        # Get the predicted class label
        predicted_class = np.argmax(predictions)
        class_name = class_labels[predicted_class]

        # Copy the image to the corresponding class folder
        shutil.copy(img_path, output_folder_path / class_name / filename)

print("Classification completed.")
