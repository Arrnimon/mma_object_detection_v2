# Quick Start Guide

## Prerequisites

1. **Install Python 3.8 or higher**
   - Download from: https://python.org
   - **Important**: Check "Add Python to PATH" during installation

2. **Install Git (optional)**
   - Download from: https://git-scm.com

## Installation

### Option 1: Automatic Installation (Windows)
1. Double-click `install.bat`
2. Follow the prompts
3. Wait for installation to complete

### Option 2: Manual Installation
1. Open Command Prompt or PowerShell
2. Navigate to the project directory
3. Run: `python -m pip install -r requirements.txt`

## Quick Usage

### GUI Application (Easiest)
```bash
python gui_detector.py
```
- Click "Browse" to select your MP4 video
- Adjust settings as needed
- Click "Detect Objects"

### Command Line (Fast)
```bash
# Basic detection
python object_detector.py your_video.mp4

# Save output video
python object_detector.py your_video.mp4 --output detected.mp4

# Detect specific objects
python object_detector.py your_video.mp4 --classes person car
```

### Batch Processing (Multiple Videos)
```bash
python batch_processor.py input_folder output_folder
```

## Example Commands

```bash
# Detect people and cars in traffic video
python object_detector.py traffic.mp4 --classes person car --confidence 0.6

# Analyze video without saving
python object_detector.py wildlife.mp4 --analyze

# Save frames with detections
python object_detector.py security.mp4 --save-frames --output-dir frames/
```

## Supported Video Formats
- MP4 (recommended)
- AVI
- MOV
- MKV
- WMV
- FLV

## Common Object Classes
- person, car, dog, cat, chair, table
- bottle, cup, bowl, book, phone
- And 70+ more from COCO dataset

## Troubleshooting

### "Python not found"
- Install Python from https://python.org
- Make sure to check "Add Python to PATH"

### "pip not found"
- Use: `python -m pip install -r requirements.txt`

### "Video file not found"
- Check the file path is correct
- Ensure the video file exists

### Slow processing
- Reduce video resolution
- Increase confidence threshold
- Use GPU if available

## Need Help?

1. Check the full README.md for detailed instructions
2. Look at example.py for code examples
3. Try the GUI application for easiest usage 