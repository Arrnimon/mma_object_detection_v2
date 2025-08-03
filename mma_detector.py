#!/usr/bin/env python3
"""
MMA Fighter Detection System
Specialized object detector for MMA fighters in octagon
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import argparse
from pathlib import Path
import time
import json

class MMADetector:
    def __init__(self, model_path=None):
        """
        Initialize MMA fighter detector
        
        Args:
            model_path (str): Path to custom trained MMA model
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Loaded custom MMA model: {model_path}")
        else:
            # Use pre-trained YOLOv8n model as base
            self.model = YOLO('yolov8n.pt')
            print("⚠️  Using pre-trained YOLOv8n model (not specialized for MMA)")
            print("💡 Train a custom model for better MMA fighter detection")
        
        # MMA-specific classes
        self.mma_classes = {
            0: 'fighter_1',      # First fighter
            1: 'fighter_2',      # Second fighter
            2: 'referee',        # Referee
            3: 'octagon',        # Octagon cage
            4: 'corner',         # Corner post
            5: 'canvas',         # Canvas/floor
            6: 'equipment'       # Equipment (gloves, mouthguard, etc.)
        }
        
        # Use custom classes if available, otherwise use COCO classes
        if hasattr(self.model, 'names') and len(self.model.names) > 0:
            self.classes = self.model.names
        else:
            self.classes = self.mma_classes
    
    def detect_mma_fighters(self, video_path, output_path=None, confidence=0.5, 
                           track_fighters=False, save_analysis=False):
        """
        Detect MMA fighters in video
        
        Args:
            video_path (str): Path to MMA video
            output_path (str): Output video path
            confidence (float): Detection confidence threshold
            track_fighters (bool): Track individual fighters across frames
            save_analysis (bool): Save detailed analysis
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 MMA Video Analysis:")
        print(f"   FPS: {fps}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Duration: {total_frames/fps:.2f} seconds")
        
        # Setup video writer
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Fighter tracking variables
        fighter_tracks = {}
        frame_count = 0
        detection_stats = {
            'fighter_1_detections': 0,
            'fighter_2_detections': 0,
            'referee_detections': 0,
            'octagon_detections': 0,
            'total_frames_with_fighters': 0
        }
        
        print("🥊 Starting MMA fighter detection...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}", end='\r')
            
            # Run detection
            results = self.model(frame, conf=confidence, verbose=False)
            
            # Process detections
            annotated_frame = frame.copy()
            frame_detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.classes.get(cls, f"class_{cls}")
                        
                        # MMA-specific processing
                        if 'fighter' in class_name.lower() or 'person' in class_name.lower():
                            # Track fighters
                            if track_fighters:
                                self._track_fighter(frame_count, (x1, y1, x2, y2), class_name, conf)
                            
                            # Update stats
                            if 'fighter_1' in class_name or cls == 0:
                                detection_stats['fighter_1_detections'] += 1
                            elif 'fighter_2' in class_name or cls == 1:
                                detection_stats['fighter_2_detections'] += 1
                            
                            detection_stats['total_frames_with_fighters'] += 1
                        
                        elif 'referee' in class_name.lower():
                            detection_stats['referee_detections'] += 1
                        
                        elif 'octagon' in class_name.lower():
                            detection_stats['octagon_detections'] += 1
                        
                        # Draw bounding box with MMA-specific colors
                        color = self._get_mma_color(class_name, cls)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        label = f"{class_name} {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        # Store detection info
                        frame_detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2),
                            'frame': frame_count,
                            'timestamp': frame_count / fps
                        })
            
            # Write frame
            if video_writer:
                video_writer.write(annotated_frame)
        
        # Clean up
        cap.release()
        if video_writer:
            video_writer.release()
        
        print(f"\n✅ MMA detection completed!")
        print(f"📊 Detection Statistics:")
        print(f"   Fighter 1 detections: {detection_stats['fighter_1_detections']}")
        print(f"   Fighter 2 detections: {detection_stats['fighter_2_detections']}")
        print(f"   Referee detections: {detection_stats['referee_detections']}")
        print(f"   Octagon detections: {detection_stats['octagon_detections']}")
        print(f"   Frames with fighters: {detection_stats['total_frames_with_fighters']}")
        
        # Save analysis if requested
        if save_analysis:
            self._save_analysis(detection_stats, fighter_tracks, video_path)
        
        return detection_stats
    
    def _track_fighter(self, frame_num, bbox, class_name, confidence):
        """Track individual fighters across frames"""
        # Simple tracking based on position and class
        fighter_id = class_name
        
        if fighter_id not in fighter_tracks:
            fighter_tracks[fighter_id] = []
        
        fighter_tracks[fighter_id].append({
            'frame': frame_num,
            'bbox': bbox,
            'confidence': confidence
        })
    
    def _get_mma_color(self, class_name, class_id):
        """Get MMA-specific colors for different classes"""
        mma_colors = {
            'fighter_1': (0, 255, 0),      # Green for fighter 1
            'fighter_2': (255, 0, 0),      # Red for fighter 2
            'referee': (0, 0, 255),        # Blue for referee
            'octagon': (255, 255, 0),      # Yellow for octagon
            'corner': (255, 0, 255),       # Magenta for corners
            'canvas': (128, 128, 128),     # Gray for canvas
            'equipment': (0, 255, 255),    # Cyan for equipment
            'person': (0, 255, 0)          # Default green for people
        }
        
        # Try to match class name
        for key, color in mma_colors.items():
            if key in class_name.lower():
                return color
        
        # Fallback to class ID based color
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
    
    def _save_analysis(self, stats, tracks, video_path):
        """Save detailed analysis to JSON file"""
        analysis = {
            'video_path': video_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': stats,
            'fighter_tracks': tracks
        }
        
        output_file = f"mma_analysis_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"📄 Analysis saved to: {output_file}")
    
    def analyze_fight_activity(self, video_path, confidence=0.5):
        """
        Analyze fight activity and engagement
        """
        print("🥊 Analyzing fight activity...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        activity_data = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Analyzing frame {frame_count}/{total_frames}", end='\r')
            
            results = self.model(frame, conf=confidence, verbose=False)
            
            frame_activity = {
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'fighters_detected': 0,
                'referee_detected': False,
                'engagement_level': 0
            }
            
            fighter_positions = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = self.classes.get(cls, f"class_{cls}")
                        
                        if 'fighter' in class_name.lower() or 'person' in class_name.lower():
                            frame_activity['fighters_detected'] += 1
                            
                            # Get fighter position (center of bounding box)
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            fighter_positions.append((center_x, center_y))
                        
                        elif 'referee' in class_name.lower():
                            frame_activity['referee_detected'] = True
            
            # Calculate engagement level based on fighter proximity
            if len(fighter_positions) >= 2:
                # Calculate distance between fighters
                dist = np.sqrt((fighter_positions[0][0] - fighter_positions[1][0])**2 + 
                             (fighter_positions[0][1] - fighter_positions[1][1])**2)
                
                # Normalize distance (closer = higher engagement)
                max_dist = np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
                engagement = 1 - (dist / max_dist)
                frame_activity['engagement_level'] = engagement
            
            activity_data.append(frame_activity)
        
        cap.release()
        
        print(f"\n✅ Fight activity analysis completed!")
        
        # Calculate summary statistics
        total_frames_with_fighters = sum(1 for d in activity_data if d['fighters_detected'] >= 2)
        avg_engagement = np.mean([d['engagement_level'] for d in activity_data if d['engagement_level'] > 0])
        
        print(f"📊 Fight Activity Summary:")
        print(f"   Frames with 2+ fighters: {total_frames_with_fighters}")
        print(f"   Average engagement level: {avg_engagement:.3f}")
        print(f"   Total analysis frames: {len(activity_data)}")
        
        return {
            'activity_data': activity_data,
            'summary': {
                'total_frames_with_fighters': total_frames_with_fighters,
                'average_engagement': avg_engagement,
                'total_frames': len(activity_data)
            }
        }

def main():
    parser = argparse.ArgumentParser(description='MMA Fighter Detection System')
    parser.add_argument('video_path', help='Path to MMA video file')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--model', '-m', help='Path to custom trained MMA model')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, 
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--track', action='store_true', 
                       help='Track individual fighters')
    parser.add_argument('--analyze', action='store_true', 
                       help='Analyze fight activity')
    parser.add_argument('--save-analysis', action='store_true', 
                       help='Save detailed analysis')
    
    args = parser.parse_args()
    
    # Initialize MMA detector
    detector = MMADetector(model_path=args.model)
    
    if args.analyze:
        # Analyze fight activity
        analysis = detector.analyze_fight_activity(args.video_path, args.confidence)
    else:
        # Detect fighters
        detector.detect_mma_fighters(
            video_path=args.video_path,
            output_path=args.output,
            confidence=args.confidence,
            track_fighters=args.track,
            save_analysis=args.save_analysis
        )

if __name__ == "__main__":
    main() 