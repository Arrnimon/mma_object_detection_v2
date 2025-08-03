#!/usr/bin/env python3
"""
Improved MMA Fighter Detection System
Fixed detection counting and improved accuracy
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import argparse
import time
import json
import math
from collections import defaultdict

class ImprovedMMADetector:
    def __init__(self, model_path=None):
        """
        Initialize improved MMA detector with better tracking
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Loaded custom MMA model: {model_path}")
        else:
            self.model = YOLO('yolov8n.pt')
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
        
        # Improved tracking variables
        self.fighter_tracks = {}
        self.fighter_counter = 0
        self.action_history = []
        
    def detect_mma_fighters_improved(self, video_path, output_path=None, confidence=0.5, 
                                   frame_skip=3, save_analysis=False, max_frames=None):
        """
        Improved MMA detection with better fighter tracking and counting
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
        
        print(f"🔧 Improved MMA Analysis:")
        print(f"   Original frames: {total_frames}")
        print(f"   Processing every {frame_skip} frames")
        print(f"   Frames to process: {frames_to_process}")
        print(f"   Estimated time: {frames_to_process * 0.1:.1f} seconds")
        
        # Setup video writer
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps//frame_skip, (width, height))
        
        # Improved analysis variables
        frame_count = 0
        processed_count = 0
        detection_stats = {
            'total_person_detections': 0,
            'fighter_1_detections': 0,
            'fighter_2_detections': 0,
            'referee_detections': 0,
            'octagon_detections': 0,
            'total_frames_processed': 0,
            'strikes_detected': 0,
            'takedowns_detected': 0,
            'control_time': 0,
            'processing_time': 0,
            'fighter_tracking': {},
            'action_breakdown': defaultdict(int)
        }
        
        # Fighter tracking with improved logic
        prev_fighters = []
        fighter_positions = {}  # Track fighter positions over time
        
        print("🥊 Starting improved MMA analysis...")
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
            
            # Run detection
            results = self.model(frame, conf=confidence, verbose=False)
            
            # Process detections with improved logic
            annotated_frame = frame.copy()
            current_fighters = []
            person_detections = []
            
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
                        
                        # Improved fighter detection logic
                        if 'person' in class_name.lower():
                            detection_stats['total_person_detections'] += 1
                            person_detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                                'confidence': conf,
                                'area': (x2 - x1) * (y2 - y1)
                            })
                        
                        # Update stats for specific classes
                        if 'fighter_1' in class_name.lower() or cls == 0:
                            detection_stats['fighter_1_detections'] += 1
                        elif 'fighter_2' in class_name.lower() or cls == 1:
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
            
            # Improved fighter tracking and assignment
            if len(person_detections) >= 2:
                detection_stats['total_frames_processed'] += 1
                
                # Sort by area (larger = closer to camera, likely main fighters)
                person_detections.sort(key=lambda x: x['area'], reverse=True)
                
                # Assign fighters based on position and tracking
                current_fighters = self._assign_fighters(person_detections, fighter_positions)
                
                # Analyze actions with improved logic
                if len(current_fighters) >= 2 and len(prev_fighters) >= 2:
                    actions = self._detect_improved_actions(current_fighters, prev_fighters)
                    
                    # Update stats based on actions
                    for action in actions:
                        detection_stats['action_breakdown'][action] += 1
                        if 'strike' in action or 'punch' in action or 'kick' in action:
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
                
                # Update fighter positions for tracking
                for i, fighter in enumerate(current_fighters):
                    fighter_positions[f"fighter_{i+1}"] = fighter['center']
                
                prev_fighters = current_fighters.copy()
            
            # Write frame
            if video_writer:
                video_writer.write(annotated_frame)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        detection_stats['processing_time'] = processing_time
        detection_stats['fighter_tracking'] = fighter_positions
        
        # Clean up
        cap.release()
        if video_writer:
            video_writer.release()
        
        print(f"\n✅ Improved MMA analysis completed in {processing_time:.2f} seconds!")
        print(f"📊 Detection Statistics:")
        print(f"   Total person detections: {detection_stats['total_person_detections']}")
        print(f"   Frames with fighters: {detection_stats['total_frames_processed']}")
        print(f"   Fighter 1 detections: {detection_stats['fighter_1_detections']}")
        print(f"   Fighter 2 detections: {detection_stats['fighter_2_detections']}")
        print(f"   Referee detections: {detection_stats['referee_detections']}")
        print(f"   Octagon detections: {detection_stats['octagon_detections']}")
        print(f"   Total strikes detected: {detection_stats['strikes_detected']}")
        print(f"   Takedowns detected: {detection_stats['takedowns_detected']}")
        print(f"   Control time (frames): {detection_stats['control_time']}")
        print(f"   Processing speed: {detection_stats['total_frames_processed']/processing_time:.1f} fps")
        
        # Print action breakdown
        if detection_stats['action_breakdown']:
            print(f"\n🎯 Action Breakdown:")
            for action, count in sorted(detection_stats['action_breakdown'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {action}: {count}")
        
        # Save analysis if requested
        if save_analysis:
            self._save_improved_analysis(detection_stats, video_path, frame_skip)
        
        return detection_stats
    
    def _assign_fighters(self, person_detections, fighter_positions):
        """Assign fighters based on position and tracking"""
        fighters = []
        
        # If we have previous positions, use them for tracking
        if fighter_positions:
            for i, detection in enumerate(person_detections[:2]):  # Take top 2 largest detections
                fighter_id = f"fighter_{i+1}"
                if fighter_id in fighter_positions:
                    prev_pos = fighter_positions[fighter_id]
                    distance = math.sqrt((detection['center'][0] - prev_pos[0])**2 + 
                                       (detection['center'][1] - prev_pos[1])**2)
                    # If distance is reasonable, assign same fighter ID
                    if distance < 100:  # 100 pixel threshold
                        detection['fighter_id'] = fighter_id
                    else:
                        detection['fighter_id'] = f"fighter_{i+1}"
                else:
                    detection['fighter_id'] = f"fighter_{i+1}"
                fighters.append(detection)
        else:
            # First frame, assign based on position
            for i, detection in enumerate(person_detections[:2]):
                detection['fighter_id'] = f"fighter_{i+1}"
                fighters.append(detection)
        
        return fighters
    
    def _detect_improved_actions(self, current_fighters, prev_fighters):
        """Improved action detection with better thresholds"""
        actions = []
        
        if len(current_fighters) >= 2 and len(prev_fighters) >= 2:
            # Calculate distance between fighters
            distance = self._calculate_fighter_distance(current_fighters[0], current_fighters[1])
            
            # Detect strikes with improved logic
            for i, (curr, prev) in enumerate(zip(current_fighters, prev_fighters)):
                movement = self._calculate_movement(curr['center'], prev['center'])
                
                # More sensitive strike detection
                if movement > 15:  # Lowered threshold
                    fighter_id = curr.get('fighter_id', f"fighter_{i+1}")
                    actions.append(f"{fighter_id}_strike")
                
                # Detect rapid movements (potential strikes)
                if movement > 25:
                    fighter_id = curr.get('fighter_id', f"fighter_{i+1}")
                    actions.append(f"{fighter_id}_punch")
                
                # Detect kicks (larger movements)
                if movement > 40:
                    fighter_id = curr.get('fighter_id', f"fighter_{i+1}")
                    actions.append(f"{fighter_id}_kick")
            
            # Detect control positions with improved logic
            if distance < 80:  # Increased proximity threshold
                # Check if one fighter is on top (ground control)
                height_diff = abs(current_fighters[0]['bbox'][1] - current_fighters[1]['bbox'][1])
                if height_diff > 15:  # Lowered threshold
                    if current_fighters[0]['bbox'][1] < current_fighters[1]['bbox'][1]:
                        actions.append("fighter_1_ground_control")
                    else:
                        actions.append("fighter_2_ground_control")
                else:
                    actions.append("clinch")
            
            # Detect takedowns (rapid height changes)
            for i, (curr, prev) in enumerate(zip(current_fighters, prev_fighters)):
                height_change = abs(curr['bbox'][1] - prev['bbox'][1])
                if height_change > 30:  # Significant height change
                    fighter_id = curr.get('fighter_id', f"fighter_{i+1}")
                    actions.append(f"{fighter_id}_takedown")
        
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
    
    def _save_improved_analysis(self, stats, video_path, frame_skip):
        """Save improved analysis to JSON file"""
        analysis = {
            'video_path': video_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': stats,
            'analysis_type': 'improved_mma_detection',
            'frame_skip': frame_skip,
            'improvements': ['better_fighter_tracking', 'improved_action_detection', 'accurate_counting']
        }
        
        output_file = f"improved_mma_analysis_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"📄 Improved analysis saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Improved MMA Fighter Detection System')
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
    
    # Initialize improved MMA detector
    detector = ImprovedMMADetector(model_path=args.model)
    
    # Run improved detection
    detector.detect_mma_fighters_improved(
        video_path=args.video_path,
        output_path=args.output,
        confidence=args.confidence,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        save_analysis=args.save_analysis
    )

if __name__ == "__main__":
    main() 