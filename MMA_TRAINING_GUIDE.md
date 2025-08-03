# 🥊 MMA Fighter Detection Training Guide

## Overview

This guide will teach you how to train a custom YOLO model to detect MMA fighters in an octagon. The model will be able to identify:
- **Fighter 1** (Red corner)
- **Fighter 2** (Blue corner) 
- **Referee**
- **Octagon cage**
- **Corner posts**
- **Canvas/floor**
- **Equipment** (gloves, mouthguard, etc.)

---

## 📋 Prerequisites

### 1. **Install Training Dependencies**
```bash
py -m pip install ultralytics roboflow supervision
```

### 2. **Required Files Structure**
```
mma_training/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── config.yaml
└── train_mma_model.py
```

---

## 🎯 Step 1: Data Collection

### **Option A: Manual Data Collection**
1. **Record MMA fights** from various angles
2. **Extract frames** at different time intervals
3. **Capture diverse scenarios**:
   - Standing exchanges
   - Ground fighting
   - Clinch positions
   - Referee interventions
   - Different lighting conditions

### **Option B: Use Existing MMA Footage**
- UFC fight highlights
- Bellator fights
- Local MMA events
- Training footage

### **Recommended Dataset Size**
- **Minimum**: 500-1000 annotated images
- **Optimal**: 2000-5000 annotated images
- **Images per class**: At least 100-200 per fighter class

---

## 🏷️ Step 2: Data Annotation

### **Annotation Tools**
1. **LabelImg** (Free, desktop)
2. **Roboflow** (Online, collaborative)
3. **CVAT** (Free, web-based)
4. **VGG Image Annotator** (Free, web-based)

### **Annotation Guidelines**

#### **Fighter Annotation**
```
- Draw bounding box around entire fighter
- Include gloves, headgear, and equipment
- Label as "fighter_1" or "fighter_2"
- Ensure consistent labeling across frames
```

#### **Referee Annotation**
```
- Draw box around referee
- Include referee's entire body
- Label as "referee"
```

#### **Octagon Annotation**
```
- Draw box around visible octagon cage
- Include corner posts if visible
- Label as "octagon" or "corner"
```

#### **Equipment Annotation**
```
- Draw boxes around visible equipment
- Gloves, mouthguards, protective gear
- Label as "equipment"
```

### **YOLO Format Labels**
Each image needs a corresponding `.txt` file:
```
# Format: class_id x_center y_center width height
# All values normalized to 0-1

0 0.5 0.6 0.3 0.4    # fighter_1
1 0.7 0.5 0.25 0.35  # fighter_2
2 0.3 0.4 0.15 0.25  # referee
3 0.1 0.1 0.8 0.8    # octagon
```

---

## ⚙️ Step 3: Dataset Configuration

### **Create `config.yaml`**
```yaml
# MMA Dataset Configuration
path: ./dataset  # Dataset root directory
train: train/images  # Train images
val: val/images      # Validation images
test: test/images    # Test images (optional)

# Classes
nc: 7  # Number of classes
names:
  0: fighter_1
  1: fighter_2
  2: referee
  3: octagon
  4: corner
  5: canvas
  6: equipment
```

---

## 🚀 Step 4: Training Script

### **Create `train_mma_model.py`**
```python
#!/usr/bin/env python3
"""
MMA Fighter Detection Model Training
"""

from ultralytics import YOLO
import os

def train_mma_model():
    """Train custom MMA fighter detection model"""
    
    # Initialize model
    model = YOLO('yolov8n.pt')  # Start with pre-trained model
    
    # Training parameters
    training_args = {
        'data': 'config.yaml',           # Dataset config file
        'epochs': 100,                   # Number of training epochs
        'imgsz': 640,                    # Input image size
        'batch': 16,                     # Batch size
        'device': 'auto',                # Use available GPU/CPU
        'workers': 4,                    # Number of worker threads
        'patience': 20,                  # Early stopping patience
        'save': True,                    # Save best model
        'save_period': 10,               # Save every N epochs
        'cache': False,                  # Cache images for faster training
        'project': 'mma_detection',      # Project name
        'name': 'mma_fighter_model',     # Experiment name
        'exist_ok': True,                # Overwrite existing experiment
        'pretrained': True,              # Use pretrained weights
        'optimizer': 'auto',             # Optimizer (SGD, Adam, etc.)
        'lr0': 0.01,                     # Initial learning rate
        'lrf': 0.01,                     # Final learning rate
        'momentum': 0.937,               # SGD momentum/Adam beta1
        'weight_decay': 0.0005,          # Optimizer weight decay
        'warmup_epochs': 3.0,            # Warmup epochs
        'warmup_momentum': 0.8,          # Warmup initial momentum
        'warmup_bias_lr': 0.1,           # Warmup initial bias lr
        'box': 7.5,                      # Box loss gain
        'cls': 0.5,                      # Class loss gain
        'dfl': 1.5,                      # DFL loss gain
        'pose': 12.0,                    # Pose loss gain
        'kobj': 2.0,                     # Keypoint obj loss gain
        'label_smoothing': 0.0,          # Label smoothing epsilon
        'nbs': 64,                       # Nominal batch size
        'overlap_mask': True,            # Masks should overlap during training
        'mask_ratio': 4,                 # Mask downsample ratio
        'dropout': 0.0,                  # Use dropout regularization
        'val': True,                     # Validate during training
        'plots': True,                   # Save plots
    }
    
    # Start training
    print("🥊 Starting MMA fighter detection model training...")
    print(f"📊 Training parameters: {training_args}")
    
    results = model.train(**training_args)
    
    print("✅ Training completed!")
    print(f"📁 Model saved to: {results.save_dir}")
    
    return results

def validate_model(model_path):
    """Validate trained model on test set"""
    model = YOLO(model_path)
    
    # Run validation
    results = model.val()
    
    print("📊 Validation Results:")
    print(f"   mAP50: {results.box.map50:.3f}")
    print(f"   mAP50-95: {results.box.map:.3f}")
    print(f"   Precision: {results.box.mp:.3f}")
    print(f"   Recall: {results.box.mr:.3f}")
    
    return results

if __name__ == "__main__":
    # Train the model
    results = train_mma_model()
    
    # Validate the model
    model_path = results.save_dir + "/weights/best.pt"
    validate_model(model_path)
```

---

## 📊 Step 5: Training Process

### **Training Commands**
```bash
# Basic training
py train_mma_model.py

# Training with custom parameters
py -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='config.yaml', epochs=100, imgsz=640, batch=16)
"
```

### **Training Monitoring**
- **Loss curves**: Monitor training and validation loss
- **mAP scores**: Mean Average Precision
- **Precision/Recall**: Detection accuracy
- **Class-wise metrics**: Performance per fighter class

### **Expected Training Time**
- **YOLOv8n**: 2-4 hours (GPU), 8-12 hours (CPU)
- **YOLOv8s**: 4-8 hours (GPU), 16-24 hours (CPU)
- **YOLOv8m**: 8-16 hours (GPU), 24-48 hours (CPU)

---

## 🎯 Step 6: Model Evaluation

### **Test on Sample Videos**
```python
from mma_detector import MMADetector

# Load trained model
detector = MMADetector(model_path='mma_detection/mma_fighter_model/weights/best.pt')

# Test on MMA video
results = detector.detect_mma_fighters(
    video_path='test_fight.mp4',
    output_path='output_fight.mp4',
    confidence=0.5,
    track_fighters=True
)
```

### **Performance Metrics**
- **Detection Accuracy**: How well fighters are detected
- **Tracking Consistency**: Fighter identification across frames
- **False Positives**: Incorrect detections
- **False Negatives**: Missed detections

---

## 🔧 Step 7: Model Optimization

### **Hyperparameter Tuning**
```python
# Experiment with different parameters
configs = [
    {'lr0': 0.01, 'epochs': 100},
    {'lr0': 0.005, 'epochs': 150},
    {'lr0': 0.02, 'epochs': 80},
    {'imgsz': 512, 'batch': 32},
    {'imgsz': 768, 'batch': 8},
]
```

### **Data Augmentation**
```yaml
# Add to config.yaml
augment:
  hsv_h: 0.015  # HSV-Hue augmentation
  hsv_s: 0.7    # HSV-Saturation augmentation
  hsv_v: 0.4    # HSV-Value augmentation
  degrees: 0.0  # Image rotation
  translate: 0.1 # Image translation
  scale: 0.5    # Image scaling
  shear: 0.0    # Image shear
  perspective: 0.0 # Perspective transform
  flipud: 0.0   # Vertical flip
  fliplr: 0.5   # Horizontal flip
  mosaic: 1.0   # Mosaic augmentation
  mixup: 0.0    # Mixup augmentation
```

---

## 🚀 Step 8: Deployment

### **Convert to Production Model**
```python
# Export model for deployment
model = YOLO('mma_detection/mma_fighter_model/weights/best.pt')
model.export(format='onnx')  # For ONNX runtime
model.export(format='tflite')  # For TensorFlow Lite
model.export(format='coreml')  # For iOS
```

### **Integration with MMA Detector**
```python
# Use trained model
detector = MMADetector(model_path='best.pt')
detector.detect_mma_fighters('fight_video.mp4')
```

---

## 📈 Advanced Features

### **Fighter Tracking**
- **Re-identification**: Track fighters across camera cuts
- **Pose estimation**: Analyze fighting stances
- **Action recognition**: Identify strikes, takedowns, submissions

### **Fight Analysis**
- **Engagement metrics**: Time spent in close combat
- **Position tracking**: Octagon position analysis
- **Strike counting**: Automated strike detection
- **Round timing**: Automatic round detection

### **Real-time Processing**
- **Live streaming**: Real-time fight analysis
- **Broadcast integration**: Overlay detection on live feeds
- **Mobile deployment**: On-device processing

---

## 🐛 Troubleshooting

### **Common Issues**

1. **Low Detection Accuracy**
   - Increase dataset size
   - Improve annotation quality
   - Adjust confidence thresholds
   - Use data augmentation

2. **Overfitting**
   - Reduce model complexity
   - Increase regularization
   - Add more validation data
   - Use early stopping

3. **Slow Training**
   - Use GPU acceleration
   - Reduce batch size
   - Use smaller model (YOLOv8n)
   - Enable image caching

4. **Poor Fighter Distinction**
   - Improve class balance
   - Add more diverse training data
   - Use tracking algorithms
   - Implement re-identification

---

## 📚 Resources

### **Datasets**
- **Roboflow MMA Dataset**: Pre-annotated MMA images
- **UFC Fight Footage**: Official fight highlights
- **Custom Collection**: Your own annotated data

### **Tools**
- **LabelImg**: https://github.com/tzutalin/labelImg
- **Roboflow**: https://roboflow.com/
- **CVAT**: https://cvat.org/

### **Documentation**
- **YOLO Training**: https://docs.ultralytics.com/modes/train/
- **Custom Datasets**: https://docs.ultralytics.com/datasets/
- **Model Export**: https://docs.ultralytics.com/modes/export/

---

## 🎯 Next Steps

1. **Start with small dataset** (100-200 images)
2. **Train initial model** and test performance
3. **Iterate and improve** based on results
4. **Expand dataset** with more diverse scenarios
5. **Deploy and monitor** in real-world conditions

---

**Happy Training! 🥊💪** 