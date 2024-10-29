from ultralytics import YOLO
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / r'models' / r'NSFW_YOLOv8.pt'
model = YOLO(model_path)

# Specify the path to the folder containing the test images
image_folder_path = BASE_DIR / r'test_image'

# Specify the path to the folder where you want to save the classified images
output_folder_path = BASE_DIR / r'output' / r'classfied_NSFW-YOLO'
# If you want to use with your image you can edit this line
# output_folder_path = (r"c:\Users\jiras\Pictures")
output_folder_path.mkdir(parents=True, exist_ok=True)

# Run inference 
results = model(
    source=image_folder_path,
    save=True,                 
    save_txt=True,             
    project=output_folder_path,  
    name='classified_results',  
    exist_ok=True              
)

print("Inference completed. Results saved to:", output_folder_path / 'classified_results')