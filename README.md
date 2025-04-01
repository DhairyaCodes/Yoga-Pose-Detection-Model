# YogZen-Model

This repository contains the yoga pose detection model used in **[YogZen](https://github.com/DhairyaCodes/Yogzen_app)**, a wellness app designed to help users practice yoga with AI-powered guidance. YogZen leverages pose recognition technology to provide real-time feedback and gamify the yoga experience.

## Model Details
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Shape**: (256, 256, 3)
- **Accuracy Achieved**: 96.81%

## Detected Yoga Poses
The model is trained to recognize the following yoga poses:
- **Downdog**
- **Goddess**
- **Plank**
- **Tree**
- **Warrior**

## Usage
To use this model for inference or fine-tuning:

```python
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the trained model
model = load_model("yogzen_model.h5")

# Preprocess image
img = cv2.imread("path_to_image.jpg")
img = cv2.resize(img, (256, 256))
img = img / 255.0  # Normalize
img = np.expand_dims(img, axis=0)

# Make prediction
predictions = model.predict(img)
class_idx = np.argmax(predictions)
classes = ['downdog', 'goddess', 'plank', 'tree', 'warrior']
print(f"Predicted Pose: {classes[class_idx]}")
```

---
Feel free to contribute or use this model for your own yoga-based applications!

