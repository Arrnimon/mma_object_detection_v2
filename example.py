#!/usr/bin/env python3
"""
Example script demonstrating how to use the VideoObjectDetector class
"""

from object_detector import VideoObjectDetector
import os
from pathlib import Path

def main():
    # Initialize the object detector
    print("Initializing object detector...")
    detector = VideoObjectDetector()
    
    # ============================================================================
    # VIDEO PATH EXAMPLES - Choose one of these methods:
    # ============================================================================
    
    # Method 1: Absolute path (Windows)
    # video_path = r"C:\Users\arnav\Videos\my_video.mp4"
    
    # Method 2: Absolute path (Unix/Linux/Mac)
    # video_path = "/home/username/Videos/my_video.mp4"
    
    # Method 3: Relative path from current directory
    # video_path = "videos/my_video.mp4"
    
    # Method 4: Using Path object (recommended)
    # video_path = Path("videos/my_video.mp4")
    
    # Method 5: Video in the same directory as this script
    # video_path = "sample_video.mp4"
    
    # ============================================================================
    # UPDATE THIS LINE WITH YOUR ACTUAL VIDEO PATH:
    # ============================================================================
    video_path = "sample-20s.mp4"  # ← Using the sample video in this directory
    
    # ============================================================================
    # SUPPORTED VIDEO FORMATS:
    # ============================================================================
    # - MP4 (.mp4)
    # - AVI (.avi) 
    # - MOV (.mov)
    # - MKV (.mkv)
    # - WMV (.wmv)
    # - FLV (.flv)
    
    # ============================================================================
    # HOW TO FIND YOUR VIDEO PATH:
    # ============================================================================
    # 1. Right-click on your video file
    # 2. Select "Properties" 
    # 3. Copy the "Location" + filename
    # 4. Example: C:\Users\arnav\Downloads\sample.mp4
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("\n📁 Please update the video_path variable with a valid video file.")
        print("\n💡 Examples of correct video paths:")
        print("   Windows: r'C:\\Users\\arnav\\Videos\\my_video.mp4'")
        print("   Windows: 'C:/Users/arnav/Videos/my_video.mp4'")
        print("   Relative: 'videos/my_video.mp4'")
        print("   Same folder: 'sample.mp4'")
        print("\n🔍 To find your video path:")
        print("   1. Right-click on your video file")
        print("   2. Select 'Properties'")
        print("   3. Copy the full path")
        return
    
    print(f"✅ Video file found: {video_path}")
    
    # Example 1: Basic object detection
    print("\n=== Example 1: Basic Object Detection ===")
    
    print(f"Processing video: {video_path}")
    
    # Detect all objects with default settings
    stats = detector.detect_objects_in_video(
        video_path=video_path,
        output_path="output_detected.mp4",
        confidence=0.5
    )
    
    print("Detection completed!")
    if stats:
        print("Objects detected:")
        for class_name, count in stats.items():
            print(f"  {class_name}: {count}")
    
    # Example 2: Detect specific objects only
    print("\n=== Example 2: Detect Specific Objects ===")
    
    print("Detecting only people and cars...")
    
    stats = detector.detect_objects_in_video(
        video_path=video_path,
        output_path="output_people_cars.mp4",
        confidence=0.6,
        target_classes=["person", "car"]
    )
    
    print("Detection completed!")
    if stats:
        print("Objects detected:")
        for class_name, count in stats.items():
            print(f"  {class_name}: {count}")
    
    # Example 3: Analyze video without saving output
    print("\n=== Example 3: Video Analysis ===")
    
    print("Analyzing video content...")
    
    analysis = detector.analyze_video(
        video_path=video_path,
        confidence=0.5,
        target_classes=["person", "car", "dog", "cat"]
    )
    
    print(f"Video duration: {analysis['duration']:.2f} seconds")
    print(f"Total frames: {analysis['total_frames']}")
    print(f"FPS: {analysis['fps']}")
    print(f"Total detections: {len(analysis['detections'])}")
    
    # Group detections by class
    class_counts = {}
    for detection in analysis['detections']:
        class_name = detection['class']
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    
    if class_counts:
        print("\nDetections by class:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}")
    
    # Example 4: Save individual frames
    print("\n=== Example 4: Save Individual Frames ===")
    
    print("Saving frames with detections...")
    
    stats = detector.detect_objects_in_video(
        video_path=video_path,
        output_path="output_with_frames.mp4",
        confidence=0.5,
        target_classes=["person"],
        save_frames=True,
        output_dir="detected_frames"
    )
    
    print("Frame saving completed!")

if __name__ == "__main__":
    main() 