# Video Object Detection Application

This application uses YOLOv8 (You Only Look Once) to detect objects in MP4 video files. It provides both a command-line interface and a graphical user interface for easy object detection and analysis.

## Features

- **Object Detection**: Detect 80 different object classes from the COCO dataset
- **Video Processing**: Process MP4, AVI, MOV, and MKV video files
- **Customizable Settings**: Adjust confidence thresholds and target specific object classes
- **Output Options**: Save annotated videos and individual frames
- **Analysis Mode**: Analyze video statistics without saving output
- **GUI Interface**: User-friendly graphical interface
- **Command Line**: Scriptable command-line interface

## Supported Object Classes

The application can detect 80 different object classes including:
- Person, car, dog, cat, chair, table, bottle, cup, bowl
- And many more from the COCO dataset

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows 10/11 (tested on Windows)

### Setup

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import ultralytics; print('YOLOv8 installed successfully')"
   ```

## Usage

### Graphical User Interface (Recommended)

1. **Launch the GUI**:
   ```bash
   python gui_detector.py
   ```

2. **Using the GUI**:
   - Click "Browse" to select your input video file
   - Adjust confidence threshold (0.1-1.0)
   - Optionally specify target classes (e.g., "person car dog")
   - Choose whether to save individual frames
   - Click "Detect Objects" to process the video
   - Or click "Analyze Video" for statistics only

### Command Line Interface

#### Basic Usage

```bash
# Detect all objects in a video
python object_detector.py video.mp4

# Save output video
python object_detector.py video.mp4 --output detected_video.mp4

# Set confidence threshold
python object_detector.py video.mp4 --confidence 0.7

# Detect specific classes only
python object_detector.py video.mp4 --classes person car dog

# Analyze video without saving output
python object_detector.py video.mp4 --analyze
```

#### Advanced Options

```bash
# Save individual frames with detections
python object_detector.py video.mp4 --save-frames --output-dir frames/

# Combine multiple options
python object_detector.py video.mp4 \
    --output detected.mp4 \
    --confidence 0.6 \
    --classes person car \
    --save-frames \
    --output-dir output_frames/
```

### Command Line Arguments

- `video_path`: Path to input video file (required)
- `--output, -o`: Path to output video file
- `--confidence, -c`: Confidence threshold (0.1-1.0, default: 0.5)
- `--classes`: Target classes to detect (space-separated)
- `--save-frames`: Save individual frames with detections
- `--output-dir`: Directory to save frames
- `--analyze`: Analyze video and show statistics only

## Examples

### Example 1: Detect People and Cars

```bash
python object_detector.py traffic_video.mp4 \
    --output traffic_detected.mp4 \
    --classes person car \
    --confidence 0.6
```

### Example 2: Analyze Video Content

```bash
python object_detector.py wildlife_video.mp4 --analyze
```

### Example 3: Save Frames with Detections

```bash
python object_detector.py security_camera.mp4 \
    --save-frames \
    --output-dir security_frames/ \
    --classes person
```

## Output

### Video Output
- Annotated video with bounding boxes around detected objects
- Labels showing object class and confidence score
- Same resolution and frame rate as input video

### Frame Output (if enabled)
- Individual JPEG frames saved to specified directory
- Frames are saved only when objects are detected
- Filename format: `frame_XXXXXX.jpg`

### Analysis Results
- Total number of detections per class
- Video duration and frame count
- Detection timestamps

## Performance Tips

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for faster processing
2. **Model Size**: YOLOv8n is fastest, YOLOv8s/m/l/x are more accurate but slower
3. **Confidence Threshold**: Higher values reduce false positives but may miss objects
4. **Target Classes**: Specifying target classes improves speed

## Troubleshooting

### Common Issues

1. **"Video file not found"**
   - Ensure the video file path is correct
   - Check file permissions

2. **"Could not open video file"**
   - Verify the video format is supported
   - Try converting to MP4 format

3. **Slow processing**
   - Reduce video resolution
   - Increase confidence threshold
   - Use GPU if available

4. **Memory errors**
   - Process shorter video segments
   - Reduce video resolution
   - Close other applications

### GPU Support

To enable GPU acceleration:

1. Install CUDA toolkit
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## File Structure

```
object_detection_project/
├── object_detector.py      # Main detection script
├── gui_detector.py         # Graphical user interface
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Technical Details

- **Model**: YOLOv8 (You Only Look Once version 8)
- **Dataset**: COCO (Common Objects in Context)
- **Classes**: 80 object classes
- **Input**: MP4, AVI, MOV, MKV video files
- **Output**: Annotated videos and/or individual frames

## License

This project uses open-source libraries:
- YOLOv8 (Ultralytics) - MIT License
- OpenCV - Apache 2.0 License
- PyTorch - BSD License

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Test with a simple video file first
4. Check the console output for error messages 