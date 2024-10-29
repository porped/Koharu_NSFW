# Koharu_NSFW
This is an Image classification project for detecting NSFW image/content based on YOLOv8. And the TensorFlow 2 for image classify the drawing image (anime, waifu game) and the real picture (person, animal, etc.)

## Installation

Make sure Python 3.8+ is installed, then install the required package:

```bash
pip install tensorflow
pip install ultralytics
```

(Optional) GPU Support
If you have an NVIDIA GPU and want to leverage CUDA, install PyTorch with GPU support: Follow instructions at PyTorch.

```bash
Koharu-NSFW/
│
├── test_image/
│   ├── test_000.jpg
│   └── test_001.jpg
│
├── models/
│   ├── TF2_human_anime_model.h5
│   └── NSFW_YOLOv8.pt
│
├── output_anime-human/
│   ├── anime/
│   │   ├── test_000.jpg
│   │   └── test_001.jpg
│   └── human/
│       └── test_002.jpg
├── classify_anime-human.py
├── classify_YOLO_NSFW.py
└── requirements.txt
