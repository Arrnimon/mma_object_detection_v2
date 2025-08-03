#!/usr/bin/env python3
"""
Simple command-line interface for object detection
Usage: py run_detection.py "path/to/your/video.mp4"
"""

import sys
import os
from object_detector import VideoObjectDetector

def main():
    if len(sys.argv) < 2:
        print("❌ Error: Please provide a video file path")
        print("\n📖 Usage:")
        print("   py run_detection.py \"path/to/your/video.mp4\"")
        print("\n💡 Examples:")
        print("   py run_detection.py \"sample-20s.mp4\"")
        print("   py run_detection.py \"C:\\Users\\arnav\\Videos\\my_video.mp4\"")
        print("   py run_detection.py \"./videos/test.mp4\"")
        return
    
    video_path = sys.argv[1]
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        print("\n🔍 Please check:")
        print("   1. The file path is correct")
        print("   2. The file exists in the specified location")
        print("   3. You have permission to access the file")
        return
    
    print(f"✅ Video file found: {video_path}")
    
    # Initialize detector
    print("🚀 Initializing object detector...")
    detector = VideoObjectDetector()
    
    # Generate output filename
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"{video_name}_detected.mp4"
    
    print(f"🎯 Processing video...")
    print(f"📤 Output will be saved as: {output_path}")
    
    try:
        # Run detection
        stats = detector.detect_objects_in_video(
            video_path=video_path,
            output_path=output_path,
            confidence=0.5
        )
        
        print(f"\n✅ Detection completed successfully!")
        print(f"📁 Output saved as: {output_path}")
        
        if stats:
            print(f"\n📊 Detection Summary:")
            total_detections = sum(stats.values())
            print(f"   Total objects detected: {total_detections}")
            print(f"   Objects by class:")
            for class_name, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                print(f"     {class_name}: {count}")
        
    except Exception as e:
        print(f"❌ Error during detection: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Make sure the video file is not corrupted")
        print("   2. Try with a different video file")
        print("   3. Check if the video format is supported (MP4, AVI, MOV, etc.)")

if __name__ == "__main__":
    main() 