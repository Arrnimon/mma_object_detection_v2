#!/usr/bin/env python3
"""
Fast MMA Fighter Detection System
Optimized for speed with frame skipping and other optimizations
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import argparse
import time
import json
import math

class FastMMADetector:
    def __init__(self, model_path=None):
        """
        Initialize fast MMA detector with optimizations
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Loaded custom MMA model: {model_path}")
        else:
            # Use smaller model for speed
            self.model = YOLO('yolov8n.pt')  # Smallest model
            print("⚡ Using YOLOv8n (fastest model)")
        
        # MMA-specific classes
        self.mma_classes = {
            0: 'fighter_1', 1: 'fighter_2', 2: 'referee',
            3: 'octagon', 4: 'corner', 5: 'canvas', 6: 'equipment'
        }
        
        # Use custom classes if available
        if hasattr(self.model, 'names') and len(self.model.names) > 0:
            self.classes = self.model.names
        else:
            self.classes = self.mma_classes
        
        # Tracking variables
        self.fighter_tracks = {}
        self.action_history = []
        
    def detect_mma_fighters_fast(self, video_path, output_path=None, confidence=0.5, 
                               frame_skip=3, save_analysis=False, max_frames=None):
        """
        Fast MMA detection with frame skipping and optimizations
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate processing info
        frames_to_process = total_frames // frame_skip
        if max_frames:
            frames_to_process = min(frames_to_process, max_frames)
        
        print(f"⚡ Fast MMA Analysis:")
        print(f"   Original frames: {total_frames}")
        print(f"   Processing every {frame_skip} frames")
        print(f"   Frames to process: {frames_to_process}")
        print(f"   Estimated time: {frames_to_process * 0.1:.1f} seconds")
        
        # Setup video writer
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps//frame_skip, (width, height))
        
        # Analysis variables
        frame_count = 0
        processed_count = 0
        detection_stats = {
            'fighter_1_detections': 0,
            'fighter_2_detections': 0,
            'referee_detections': 0,
            'octagon_detections': 0,
            'total_frames_processed': 0,
            'strikes_detected': 0,
            'takedowns_detected': 0,
            'control_time': 0,
            'processing_time': 0
        }
        
        # Fighter tracking
        prev_fighters = []
        
        print("🥊 Starting fast MMA analysis...")
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for speed
            if frame_count % frame_skip != 0:
                continue
            
            # Limit processing if max_frames specified
            if max_frames and processed_count >= max_frames:
                break
            
            processed_count += 1
            print(f"Processing frame {processed_count}/{frames_to_process} (original: {frame_count})", end='\r')
            
            # Run detection with optimizations
            results = self.model(frame, conf=confidence, verbose=False)
            
            # Process detections
            annotated_frame = frame.copy()
            current_fighters = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        class_name = self.classes.get(cls, f"class_{cls}")
                        
                        # Store fighter info for action detection
                        if 'fighter' in class_name.lower() or 'person' in class_name.lower():
                            fighter_info = {
                                'bbox': (x1, y1, x2, y2),
                                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                                'class': class_name,
                                'confidence': conf
                            }
                            current_fighters.append(fighter_info)
                        
                        # Update stats
                        if 'fighter_1' in class_name or cls == 0:
                            detection_stats['fighter_1_detections'] += 1
                        elif 'fighter_2' in class_name or cls == 1:
                            detection_stats['fighter_2_detections'] += 1
                        elif 'referee' in class_name.lower():
                            detection_stats['referee_detections'] += 1
                        elif 'octagon' in class_name.lower():
                            detection_stats['octagon_detections'] += 1
                        
                        # Draw bounding box
                        color = self._get_mma_color(class_name, cls)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        label = f"{class_name} {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Analyze actions if we have fighters
            if len(current_fighters) >= 2 and len(prev_fighters) >= 2:
                detection_stats['total_frames_processed'] += 1
                
                # Simple action detection based on movement
                actions = self._detect_simple_actions(current_fighters, prev_fighters)
                
                # Update stats based on actions
                for action in actions:
                    if 'strike' in action:
                        detection_stats['strikes_detected'] += 1
                    elif 'takedown' in action:
                        detection_stats['takedowns_detected'] += 1
                    elif 'control' in action:
                        detection_stats['control_time'] += 1
                
                # Display actions
                if actions:
                    action_text = ", ".join(actions)
                    cv2.putText(annotated_frame, f"Actions: {action_text}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update fighter history
            prev_fighters = current_fighters.copy()
            
            # Write frame
            if video_writer:
                video_writer.write(annotated_frame)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        detection_stats['processing_time'] = processing_time
        
        # Clean up
        cap.release()
        if video_writer:
            video_writer.release()
        
        print(f"\n✅ Fast MMA analysis completed in {processing_time:.2f} seconds!")
        print(f"📊 Detection Statistics:")
        print(f"   Frames processed: {detection_stats['total_frames_processed']}")
        print(f"   Fighter 1 detections: {detection_stats['fighter_1_detections']}")
        print(f"   Fighter 2 detections: {detection_stats['fighter_2_detections']}")
        print(f"   Referee detections: {detection_stats['referee_detections']}")
        print(f"   Octagon detections: {detection_stats['octagon_detections']}")
        print(f"   Strikes detected: {detection_stats['strikes_detected']}")
        print(f"   Takedowns detected: {detection_stats['takedowns_detected']}")
        print(f"   Control time (frames): {detection_stats['control_time']}")
        print(f"   Processing speed: {detection_stats['total_frames_processed']/processing_time:.1f} fps")
        
        # Save analysis if requested
        if save_analysis:
            self._save_fast_analysis(detection_stats, video_path, frame_skip)
        
        return detection_stats
    
    def _detect_simple_actions(self, current_fighters, prev_fighters):
        """Simple action detection based on movement"""
        actions = []
        
        if len(current_fighters) >= 2 and len(prev_fighters) >= 2:
            # Calculate distance between fighters
            distance = self._calculate_fighter_distance(current_fighters[0], current_fighters[1])
            
            # Detect strikes based on movement
            for i, (curr, prev) in enumerate(zip(current_fighters, prev_fighters)):
                movement = self._calculate_movement(curr['center'], prev['center'])
                if movement > 30:  # Threshold for strike detection
                    actions.append(f"fighter_{i+1}_strike")
            
            # Detect control positions
            if distance < 50:  # Close proximity
                # Check if one fighter is on top (simple height check)
                height_diff = abs(current_fighters[0]['bbox'][1] - current_fighters[1]['bbox'][1])
                if height_diff > 20:
                    if current_fighters[0]['bbox'][1] < current_fighters[1]['bbox'][1]:
                        actions.append("fighter_1_ground_control")
                    else:
                        actions.append("fighter_2_ground_control")
                else:
                    actions.append("clinch")
        
        return actions
    
    def _calculate_fighter_distance(self, fighter1, fighter2):
        """Calculate distance between two fighters"""
        center1 = fighter1['center']
        center2 = fighter2['center']
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _calculate_movement(self, current_pos, prev_pos):
        """Calculate movement between two positions"""
        return math.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
    
    def _get_mma_color(self, class_name, class_id):
        """Get MMA-specific colors"""
        mma_colors = {
            'fighter_1': (0, 255, 0),      # Green
            'fighter_2': (255, 0, 0),      # Red
            'referee': (0, 0, 255),        # Blue
            'octagon': (255, 255, 0),      # Yellow
            'corner': (255, 0, 255),       # Magenta
            'canvas': (128, 128, 128),     # Gray
            'equipment': (0, 255, 255),    # Cyan
            'person': (0, 255, 0)          # Default green
        }
        
        for key, color in mma_colors.items():
            if key in class_name.lower():
                return color
        
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
    
    def _save_fast_analysis(self, stats, video_path, frame_skip):
        """Save fast analysis to JSON file"""
        analysis = {
            'video_path': video_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': stats,
            'analysis_type': 'fast_mma_detection',
            'frame_skip': frame_skip,
            'speed_optimizations': ['frame_skipping', 'simplified_actions', 'yolov8n_model']
        }
        
        output_file = f"fast_mma_analysis_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"📄 Fast analysis saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Fast MMA Fighter Detection System')
    parser.add_argument('video_path', help='Path to MMA video file')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--model', '-m', help='Path to custom trained MMA model')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, 
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--frame-skip', '-f', type=int, default=3, 
                       help='Process every Nth frame (default: 3)')
    parser.add_argument('--max-frames', type=int, 
                       help='Maximum frames to process (for testing)')
    parser.add_argument('--save-analysis', action='store_true', 
                       help='Save detailed analysis')
    
    args = parser.parse_args()
    
    # Initialize fast MMA detector
    detector = FastMMADetector(model_path=args.model)
    
    # Run fast detection
    detector.detect_mma_fighters_fast(
        video_path=args.video_path,
        output_path=args.output,
        confidence=args.confidence,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        save_analysis=args.save_analysis
    )

if __name__ == "__main__":
    main() 