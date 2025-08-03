#!/usr/bin/env python3
"""
Interactive Demo: Understanding Object Detection Concepts
This script demonstrates how object detection works step by step
"""

from object_detector import VideoObjectDetector
import cv2
import numpy as np
import time

def demo_basic_concepts():
    """Demonstrate basic object detection concepts"""
    print("🎯 OBJECT DETECTION CONCEPTS DEMO")
    print("=" * 50)
    
    # Initialize detector
    print("\n1️⃣ Initializing YOLO Model...")
    detector = VideoObjectDetector()
    print(f"✅ Model loaded successfully!")
    print(f"📋 Can detect {len(detector.classes)} different object types")
    
    # Show some example classes
    print(f"\n🎨 Example object classes:")
    example_classes = ['person', 'car', 'dog', 'cat', 'chair', 'tv', 'laptop']
    for i, class_name in enumerate(example_classes, 1):
        print(f"   {i}. {class_name}")
    
    print(f"\n📊 Total classes available: {len(detector.classes)}")

def demo_confidence_thresholds():
    """Demonstrate how confidence thresholds affect detection"""
    print("\n\n🎚️ CONFIDENCE THRESHOLD DEMO")
    print("=" * 50)
    
    print("\nConfidence threshold determines how certain the model must be:")
    print("• 0.3 = 30% sure (detects more objects, may include false positives)")
    print("• 0.5 = 50% sure (balanced detection - default)")
    print("• 0.8 = 80% sure (only very confident detections)")
    
    # Test with sample video
    video_path = "sample-20s.mp4"
    
    if not cv2.VideoCapture(video_path).isOpened():
        print(f"\n❌ Sample video not found: {video_path}")
        return
    
    detector = VideoObjectDetector()
    
    confidence_levels = [0.3, 0.5, 0.8]
    
    for conf in confidence_levels:
        print(f"\n🔍 Testing confidence threshold: {conf}")
        print(f"   Processing video with {conf*100}% confidence...")
        
        try:
            # Quick analysis to show differences
            analysis = detector.analyze_video(
                video_path=video_path,
                confidence=conf,
                target_classes=["person", "car"]  # Focus on common objects
            )
            
            # Count detections
            class_counts = {}
            for detection in analysis['detections']:
                class_name = detection['class']
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
            
            print(f"   Results with {conf*100}% confidence:")
            for class_name, count in class_counts.items():
                print(f"     {class_name}: {count} detections")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def demo_bounding_boxes():
    """Demonstrate bounding box concepts"""
    print("\n\n📦 BOUNDING BOX CONCEPTS")
    print("=" * 50)
    
    print("\nA bounding box is a rectangle around detected objects:")
    print("""
    (x1, y1) ┌─────────────┐
             │             │
             │   Object    │
             │             │
             └─────────────┘ (x2, y2)
    """)
    
    print("Coordinates represent:")
    print("• x1, y1 = Top-left corner")
    print("• x2, y2 = Bottom-right corner")
    print("• Width = x2 - x1")
    print("• Height = y2 - y1")
    
    print("\n🎨 Each object class gets a unique color:")
    detector = VideoObjectDetector()
    example_classes = ['person', 'car', 'dog', 'cat']
    
    for i, class_name in enumerate(example_classes):
        color = detector._get_color(i)
        print(f"   {class_name}: RGB{color}")

def demo_video_processing():
    """Demonstrate video processing concepts"""
    print("\n\n🎬 VIDEO PROCESSING CONCEPTS")
    print("=" * 50)
    
    print("\nVideos are processed frame by frame:")
    print("""
    Video File
        ↓
    Frame 1 → Detect Objects → Draw Boxes → Save Frame
        ↓
    Frame 2 → Detect Objects → Draw Boxes → Save Frame
        ↓
    Frame 3 → Detect Objects → Draw Boxes → Save Frame
        ↓
    ... (continues for all frames)
        ↓
    Output Video with Detections
    """)
    
    print("\n📊 Video Properties:")
    video_path = "sample-20s.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if cap.isOpened():
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"   FPS (Frames Per Second): {fps}")
        print(f"   Resolution: {width}x{height} pixels")
        print(f"   Total Frames: {total_frames}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Processing time estimate: {duration*2:.1f} seconds (2x real-time)")
        
        cap.release()
    else:
        print(f"   ❌ Could not read video: {video_path}")

def demo_usage_examples():
    """Show practical usage examples"""
    print("\n\n💡 PRACTICAL USAGE EXAMPLES")
    print("=" * 50)
    
    print("\n1️⃣ Basic Detection:")
    print("""
    from object_detector import VideoObjectDetector
    
    detector = VideoObjectDetector()
    stats = detector.detect_objects_in_video(
        video_path="my_video.mp4",
        output_path="output.mp4",
        confidence=0.5
    )
    """)
    
    print("\n2️⃣ Detect Specific Objects Only:")
    print("""
    stats = detector.detect_objects_in_video(
        video_path="my_video.mp4",
        output_path="people_cars.mp4",
        confidence=0.6,
        target_classes=["person", "car"]
    )
    """)
    
    print("\n3️⃣ Save Individual Frames:")
    print("""
    stats = detector.detect_objects_in_video(
        video_path="my_video.mp4",
        output_path="output.mp4",
        confidence=0.5,
        save_frames=True,
        output_dir="detected_frames/"
    )
    """)
    
    print("\n4️⃣ Analyze Without Saving Video:")
    print("""
    analysis = detector.analyze_video(
        video_path="my_video.mp4",
        confidence=0.5,
        target_classes=["person", "car", "dog"]
    )
    print(f"Found {len(analysis['detections'])} objects")
    """)

def interactive_demo():
    """Run an interactive demo with user input"""
    print("\n\n🎮 INTERACTIVE DEMO")
    print("=" * 50)
    
    print("\nLet's test object detection on your sample video!")
    
    video_path = "sample-20s.mp4"
    if not cv2.VideoCapture(video_path).isOpened():
        print(f"❌ Sample video not found: {video_path}")
        return
    
    print(f"✅ Found sample video: {video_path}")
    
    # Get user preferences
    print("\nChoose your detection settings:")
    
    # Confidence threshold
    while True:
        try:
            conf_input = input("Confidence threshold (0.1-1.0, default 0.5): ").strip()
            if conf_input == "":
                confidence = 0.5
                break
            confidence = float(conf_input)
            if 0.1 <= confidence <= 1.0:
                break
            else:
                print("❌ Please enter a value between 0.1 and 1.0")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Target classes
    print("\nAvailable object classes (enter numbers, separated by spaces):")
    detector = VideoObjectDetector()
    common_classes = ['person', 'car', 'dog', 'cat', 'chair', 'tv', 'laptop', 'book']
    
    for i, class_name in enumerate(common_classes, 1):
        print(f"   {i}. {class_name}")
    print("   0. Detect all objects")
    
    while True:
        try:
            class_input = input("Select classes (e.g., '1 2 3' or '0' for all): ").strip()
            if class_input == "0":
                target_classes = None
                break
            class_indices = [int(x) - 1 for x in class_input.split()]
            target_classes = [common_classes[i] for i in class_indices if 0 <= i < len(common_classes)]
            if target_classes:
                break
            else:
                print("❌ Please select valid class numbers")
        except (ValueError, IndexError):
            print("❌ Please enter valid numbers")
    
    print(f"\n🎯 Running detection with:")
    print(f"   Confidence: {confidence}")
    print(f"   Target classes: {target_classes if target_classes else 'All objects'}")
    
    # Run detection
    try:
        detector = VideoObjectDetector()
        
        output_path = f"demo_output_{int(time.time())}.mp4"
        
        print(f"\n🚀 Starting detection...")
        stats = detector.detect_objects_in_video(
            video_path=video_path,
            output_path=output_path,
            confidence=confidence,
            target_classes=target_classes
        )
        
        print(f"\n✅ Detection completed!")
        print(f"📁 Output saved as: {output_path}")
        
        if stats:
            print(f"\n📊 Results:")
            total_detections = sum(stats.values())
            print(f"   Total objects detected: {total_detections}")
            for class_name, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                print(f"   {class_name}: {count}")
        else:
            print("   No objects detected with current settings")
            
    except Exception as e:
        print(f"❌ Error during detection: {str(e)}")

def main():
    """Run all demos"""
    print("🎯 OBJECT DETECTION LEARNING DEMO")
    print("=" * 60)
    
    # Run concept demos
    demo_basic_concepts()
    demo_confidence_thresholds()
    demo_bounding_boxes()
    demo_video_processing()
    demo_usage_examples()
    
    # Ask if user wants interactive demo
    print("\n" + "=" * 60)
    response = input("\n🎮 Would you like to try an interactive demo? (y/n): ").strip().lower()
    
    if response in ['y', 'yes']:
        interactive_demo()
    
    print("\n🎓 Learning complete! Check the OBJECT_DETECTION_TUTORIAL.md for more details.")
    print("Happy detecting! 🎯✨")

if __name__ == "__main__":
    main() 