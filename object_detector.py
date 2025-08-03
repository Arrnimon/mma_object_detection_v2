import cv2
import numpy as np
from ultralytics import YOLO
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time


video_path = "C:/Users/arnav/object_detection_project/sample-20s.mp4"

class VideoObjectDetector:
    def __init__(self, model_path=None):
        """
        Initialize the object detector with YOLOv8 model
        
        Args:
            model_path (str): Path to custom YOLO model (optional)
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Use pre-trained YOLOv8n model
            self.model = YOLO('yolov8n.pt')
        
        # COCO dataset classes (80 classes)
        self.classes = self.model.names
        
    def detect_objects_in_video(self, video_path, output_path=None, confidence=0.5, 
                               target_classes=None, save_frames=False, output_dir=None):
        """
        Detect objects in an MP4 video file
        
        Args:
            video_path (str): Path to input video file
            output_path (str): Path to output video file (optional)
            confidence (float): Confidence threshold for detections
            target_classes (list): List of class names to detect (if None, detect all)
            save_frames (bool): Whether to save individual frames with detections
            output_dir (str): Directory to save frames (if save_frames=True)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties:")
        print(f"  FPS: {fps}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Total frames: {total_frames}")
        
        # Setup video writer if output path is provided
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Create output directory for frames if needed
        if save_frames and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        frame_count = 0
        detection_stats = {}
        
        print("Starting object detection...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}", end='\r')
            
            # Run object detection
            results = self.model(frame, conf=confidence, verbose=False)
            
            # Process detections
            annotated_frame = frame.copy()
            frame_detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get class and confidence
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        class_name = self.classes[cls]
                        
                        # Filter by target classes if specified
                        if target_classes and class_name not in target_classes:
                            continue
                        
                        # Store detection info
                        detection_info = {
                            'class': class_name,
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2)
                        }
                        frame_detections.append(detection_info)
                        
                        # Update statistics
                        if class_name not in detection_stats:
                            detection_stats[class_name] = 0
                        detection_stats[class_name] += 1
                        
                        # Draw bounding box
                        color = self._get_color(cls)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name} {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Write frame to output video
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Save individual frame if requested
            if save_frames and output_dir and frame_detections:
                frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, annotated_frame)
        
        # Clean up
        cap.release()
        if video_writer:
            video_writer.release()
        
        print(f"\nDetection completed!")
        print(f"Total frames processed: {frame_count}")
        print(f"Detection statistics:")
        for class_name, count in detection_stats.items():
            print(f"  {class_name}: {count} detections")
        
        return detection_stats
    
    def _get_color(self, class_id):
        """Generate a consistent color for each class"""
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
    
    def analyze_video(self, video_path, confidence=0.5, target_classes=None):
        """
        Analyze video and return detailed detection information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        frame_count = 0
        detection_data = []
        
        print(f"Analyzing video (duration: {duration:.2f} seconds)...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Analyzing frame {frame_count}/{total_frames}", end='\r')
            
            results = self.model(frame, conf=confidence, verbose=False)
            
            frame_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        class_name = self.classes[cls]
                        
                        if target_classes and class_name not in target_classes:
                            continue
                        
                        frame_detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'frame': frame_count,
                            'timestamp': frame_count / fps
                        })
            
            if frame_detections:
                detection_data.extend(frame_detections)
        
        cap.release()
        
        print(f"\nAnalysis completed!")
        return {
            'total_frames': total_frames,
            'duration': duration,
            'fps': fps,
            'detections': detection_data
        }

def main():
    parser = argparse.ArgumentParser(description='Detect objects in MP4 video files')
    parser.add_argument('video_path', help='Path to input MP4 video file')
    parser.add_argument('--output', '-o', help='Path to output video file')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, 
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--classes', nargs='+', help='Target classes to detect')
    parser.add_argument('--save-frames', action='store_true', 
                       help='Save individual frames with detections')
    parser.add_argument('--output-dir', help='Directory to save frames')
    parser.add_argument('--analyze', action='store_true', 
                       help='Analyze video and show statistics')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = VideoObjectDetector()
    
    if args.analyze:
        # Analyze video
        analysis = detector.analyze_video(args.video_path, args.confidence, args.classes)
        
        print(f"\nVideo Analysis Results:")
        print(f"Duration: {analysis['duration']:.2f} seconds")
        print(f"Total detections: {len(analysis['detections'])}")
        
        # Group detections by class
        class_counts = {}
        for detection in analysis['detections']:
            class_name = detection['class']
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        print(f"\nDetections by class:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}")
    
    else:
        # Detect objects in video
        detector.detect_objects_in_video(
            video_path=args.video_path,
            output_path=args.output,
            confidence=args.confidence,
            target_classes=args.classes,
            save_frames=args.save_frames,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main() 