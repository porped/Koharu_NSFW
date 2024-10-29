# Weebo_NSFW
This is an Image classification project for detecting NSFW image/content based on YOLOv8. And the TensorFlow 2 for image classify the drawing image (anime, waifu game) and the real picture (person, animal, etc.)

## Installation

Make sure Python 3.8+ is installed, then install the required package:

```bash
pip install tensorflow
pip install ultralytics
```
OR run command to install from requirements.txt
(Optional) GPU Support
If you have an NVIDIA GPU and want to leverage CUDA, install PyTorch with GPU support: Follow instructions at PyTorch.

Download the model from: https://drive.google.com/drive/folders/1x1OD9XTNj5KMoKoAUEYEBzY1Y1VK2n6z?usp=sharing

## Project Structure
```bash
Weebo-NSFW/
│
├── test_image/
│   ├── test_000.jpg
│   └── test_001.jpg
│
├── models/
│   ├── TF2_human_anime_model.h5
│   └── NSFW_YOLOv8.pt
│
├── classify_anime-human.py
├── classify_YOLO_NSFW.py
└── requirements.txt
```

## Usage
0. Download the model from Google drive link: https://drive.google.com/drive/folders/1x1OD9XTNj5KMoKoAUEYEBzY1Y1VK2n6z?usp=sharing.
1. Place all model in folder "Weebo-NSFW/models"
2. Run the following command to classify images:
```bash
python classify_anime-human.py
python classify_YOLO_NSFW.py
```
3. Or place your test images inside the test_image/ folder and run the command.
The results will be saved in the output/classified_results/ folder.
```bash
output/
│
├── classfied_anime-human/
│   │
│   ├── anime/
│   │   ├── test_000.jpg
│   │   └── test_001.jpg
│   │
│   └── human/
│       └── test_002.jpg
│
└── classfied_NSFW-YOLO/
    │
    └── classified_results/
        │
        ├── labels
        │   ├── test_000.txt
        │   └── test_001.txt
        │
        ├── test_000.jpg
        └── test_001.jpg
```

Example Output
Processed images will contain bounding boxes showing the detected content. Each image will have a corresponding .txt file with the detection results.
