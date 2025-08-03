#!/usr/bin/env python3
"""
Enhanced MMA Fighter Detection System
Combines YOLO detection with pose estimation and action recognition
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import argparse
from pathlib import Path
import time
import json
from collections import deque
import math

class PoseEstimator:
    """Simple pose estimation using OpenCV DNN"""
    
    def __init__(self):
        # COCO keypoints
        self.keypoints = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Use simple pose estimation (no external model files needed)
        self.use_simple_pose = True
        print("✅ Using simple pose estimation (no external models required)")
        
    def estimate_pose(self, frame, bbox):
        """Estimate pose for a detected fighter"""
        if self.use_simple_pose:
            return self._simple_pose_estimation(frame, bbox)
        
        x1, y1, x2, y2 = bbox
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return None
        
        # Prepare input for pose estimation
        blob = cv2.dnn.blobFromImage(person_roi, 1.0/255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()
        
        # Extract keypoints
        keypoints = []
        for i in range(len(self.keypoints)):
            prob_map = output[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(prob_map)
            
            if conf > 0.1:  # Confidence threshold
                x = point[0] * (x2 - x1) / 368 + x1
                y = point[1] * (y2 - y1) / 368 + y1
                keypoints.append((x, y, conf))
            else:
                keypoints.append(None)
        
        return keypoints
    
    def _simple_pose_estimation(self, frame, bbox):
        """Simple pose estimation using body proportions"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Estimate keypoints based on body proportions
        keypoints = []
        
        # Head (nose)
        nose_x = x1 + width * 0.5
        nose_y = y1 + height * 0.15
        keypoints.append((nose_x, nose_y, 0.8))
        
        # Shoulders
        left_shoulder_x = x1 + width * 0.3
        left_shoulder_y = y1 + height * 0.25
        keypoints.append((left_shoulder_x, left_shoulder_y, 0.7))
        
        right_shoulder_x = x1 + width * 0.7
        right_shoulder_y = y1 + height * 0.25
        keypoints.append((right_shoulder_x, right_shoulder_y, 0.7))
        
        # Arms
        left_elbow_x = x1 + width * 0.2
        left_elbow_y = y1 + height * 0.4
        keypoints.append((left_elbow_x, left_elbow_y, 0.6))
        
        right_elbow_x = x1 + width * 0.8
        right_elbow_y = y1 + height * 0.4
        keypoints.append((right_elbow_x, right_elbow_y, 0.6))
        
        # Hands
        left_wrist_x = x1 + width * 0.1
        left_wrist_y = y1 + height * 0.55
        keypoints.append((left_wrist_x, left_wrist_y, 0.5))
        
        right_wrist_x = x1 + width * 0.9
        right_wrist_y = y1 + height * 0.55
        keypoints.append((right_wrist_x, right_wrist_y, 0.5))
        
        # Hips
        left_hip_x = x1 + width * 0.4
        left_hip_y = y1 + height * 0.6
        keypoints.append((left_hip_x, left_hip_y, 0.7))
        
        right_hip_x = x1 + width * 0.6
        right_hip_y = y1 + height * 0.6
        keypoints.append((right_hip_x, right_hip_y, 0.7))
        
        # Legs
        left_knee_x = x1 + width * 0.4
        left_knee_y = y1 + height * 0.8
        keypoints.append((left_knee_x, left_knee_y, 0.6))
        
        right_knee_x = x1 + width * 0.6
        right_knee_y = y1 + height * 0.8
        keypoints.append((right_knee_x, right_knee_y, 0.6))
        
        # Feet
        left_ankle_x = x1 + width * 0.4
        left_ankle_y = y1 + height * 0.95
        keypoints.append((left_ankle_x, left_ankle_y, 0.5))
        
        right_ankle_x = x1 + width * 0.6
        right_ankle_y = y1 + height * 0.95
        keypoints.append((right_ankle_x, right_ankle_y, 0.5))
        
        return keypoints

class ActionRecognizer:
    """Recognize MMA actions based on pose and movement"""
    
    def __init__(self):
        self.action_history = deque(maxlen=10)  # Store last 10 frames
        self.strike_threshold = 50  # Minimum movement for strike detection
        self.control_threshold = 30  # Pixels for control detection
        
    def analyze_pose(self, keypoints1, keypoints2, prev_keypoints1=None, prev_keypoints2=None):
        """Analyze poses to detect actions"""
        actions = []
        
        if keypoints1 and keypoints2:
            # Calculate distances between fighters
            distance = self._calculate_fighter_distance(keypoints1, keypoints2)
            
            # Detect strikes
            if prev_keypoints1 and prev_keypoints2:
                strikes = self._detect_strikes(keypoints1, keypoints2, prev_keypoints1, prev_keypoints2)
                actions.extend(strikes)
            
            # Detect control positions
            control = self._detect_control_positions(keypoints1, keypoints2, distance)
            if control:
                actions.append(control)
            
            # Detect takedowns
            takedown = self._detect_takedown(keypoints1, keypoints2)
            if takedown:
                actions.append(takedown)
        
        return actions
    
    def _calculate_fighter_distance(self, keypoints1, keypoints2):
        """Calculate distance between two fighters"""
        if not keypoints1 or not keypoints2:
            return float('inf')
        
        # Use shoulder positions for distance calculation
        if len(keypoints1) >= 6 and len(keypoints2) >= 6:
            shoulder1 = keypoints1[5]  # Left shoulder
            shoulder2 = keypoints2[5]  # Left shoulder
            
            if shoulder1 and shoulder2:
                return math.sqrt((shoulder1[0] - shoulder2[0])**2 + (shoulder1[1] - shoulder2[1])**2)
        
        return float('inf')
    
    def _detect_strikes(self, keypoints1, keypoints2, prev_keypoints1, prev_keypoints2):
        """Detect punches and kicks based on hand/foot movement"""
        strikes = []
        
        # Check for punches (hand movement)
        for i, (curr, prev) in enumerate([(keypoints1, prev_keypoints1), (keypoints2, prev_keypoints2)]):
            if curr and prev and len(curr) >= 17 and len(prev) >= 17:
                # Check left hand movement
                if curr[9] and prev[9]:  # Left wrist
                    movement = math.sqrt((curr[9][0] - prev[9][0])**2 + (curr[9][1] - prev[9][1])**2)
                    if movement > self.strike_threshold:
                        strikes.append(f"fighter_{i+1}_punch")
                
                # Check right hand movement
                if curr[10] and prev[10]:  # Right wrist
                    movement = math.sqrt((curr[10][0] - prev[10][0])**2 + (curr[10][1] - prev[10][1])**2)
                    if movement > self.strike_threshold:
                        strikes.append(f"fighter_{i+1}_punch")
                
                # Check for kicks (foot movement)
                if curr[15] and prev[15]:  # Left ankle
                    movement = math.sqrt((curr[15][0] - prev[15][0])**2 + (curr[15][1] - prev[15][1])**2)
                    if movement > self.strike_threshold:
                        strikes.append(f"fighter_{i+1}_kick")
                
                if curr[16] and prev[16]:  # Right ankle
                    movement = math.sqrt((curr[16][0] - prev[16][0])**2 + (curr[16][1] - prev[16][1])**2)
                    if movement > self.strike_threshold:
                        strikes.append(f"fighter_{i+1}_kick")
        
        return strikes
    
    def _detect_control_positions(self, keypoints1, keypoints2, distance):
        """Detect ground control and clinch positions"""
        if distance < self.control_threshold:
            # Check if fighters are close (clinch or ground control)
            if len(keypoints1) >= 13 and len(keypoints2) >= 13:
                # Check if one fighter is on top (ground control)
                hip1_y = keypoints1[12][1] if keypoints1[12] else 0  # Left hip
                hip2_y = keypoints2[12][1] if keypoints2[12] else 0
                
                if abs(hip1_y - hip2_y) > 20:  # Significant height difference
                    if hip1_y < hip2_y:
                        return "fighter_1_ground_control"
                    else:
                        return "fighter_2_ground_control"
                else:
                    return "clinch"
        
        return None
    
    def _detect_takedown(self, keypoints1, keypoints2):
        """Detect takedown attempts"""
        if len(keypoints1) >= 13 and len(keypoints2) >= 13:
            # Check for rapid downward movement
            hip1_y = keypoints1[12][1] if keypoints1[12] else 0
            hip2_y = keypoints2[12][1] if keypoints2[12] else 0
            
            # If one fighter's hips are much lower, it might be a takedown
            if abs(hip1_y - hip2_y) > 50:
                if hip1_y > hip2_y:
                    return "fighter_1_takedown"
                else:
                    return "fighter_2_takedown"
        
        return None

class EnhancedMMADetector:
    def __init__(self, model_path=None):
        """
        Initialize enhanced MMA detector with pose estimation
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Loaded custom MMA model: {model_path}")
        else:
            self.model = YOLO('yolov8n.pt')
            print("⚠️  Using pre-trained YOLOv8n model")
        
        # Initialize pose estimator and action recognizer
        self.pose_estimator = PoseEstimator()
        self.action_recognizer = ActionRecognizer()
        
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
        self.pose_history = {}
        self.action_history = []
        
    def detect_mma_fighters_with_pose(self, video_path, output_path=None, confidence=0.5, 
                                    track_fighters=True, save_analysis=False):
        """
        Enhanced MMA detection with pose estimation and action recognition
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
        
        print(f"🎬 Enhanced MMA Analysis:")
        print(f"   FPS: {fps}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Duration: {total_frames/fps:.2f} seconds")
        
        # Setup video writer
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Analysis variables
        frame_count = 0
        detection_stats = {
            'fighter_1_detections': 0,
            'fighter_2_detections': 0,
            'referee_detections': 0,
            'octagon_detections': 0,
            'total_frames_with_fighters': 0,
            'strikes_detected': 0,
            'takedowns_detected': 0,
            'control_time': 0
        }
        
        # Pose tracking
        prev_poses = {}
        
        print("🥊 Starting enhanced MMA analysis...")
        
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
            current_poses = {}
            fighter_bboxes = {}
            
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
                        
                        # Store fighter bounding boxes for pose estimation
                        if 'fighter' in class_name.lower() or 'person' in class_name.lower():
                            fighter_id = f"fighter_{len(fighter_bboxes) + 1}"
                            fighter_bboxes[fighter_id] = (x1, y1, x2, y2)
                            
                            # Estimate pose
                            pose = self.pose_estimator.estimate_pose(frame, (x1, y1, x2, y2))
                            if pose:
                                current_poses[fighter_id] = pose
                        
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
            
            # Analyze actions if we have poses
            if len(current_poses) >= 2:
                detection_stats['total_frames_with_fighters'] += 1
                
                # Get poses for both fighters
                poses = list(current_poses.values())
                prev_poses_list = list(prev_poses.values()) if len(prev_poses) >= 2 else [None, None]
                
                # Analyze actions
                actions = self.action_recognizer.analyze_pose(
                    poses[0], poses[1], 
                    prev_poses_list[0] if len(prev_poses_list) > 0 else None,
                    prev_poses_list[1] if len(prev_poses_list) > 1 else None
                )
                
                # Update stats based on actions
                for action in actions:
                    if 'punch' in action or 'kick' in action:
                        detection_stats['strikes_detected'] += 1
                    elif 'takedown' in action:
                        detection_stats['takedowns_detected'] += 1
                    elif 'control' in action:
                        detection_stats['control_time'] += 1
                
                # Draw pose keypoints
                for fighter_id, pose in current_poses.items():
                    if pose:
                        self._draw_pose_keypoints(annotated_frame, pose, fighter_id)
                
                # Display actions
                if actions:
                    action_text = ", ".join(actions)
                    cv2.putText(annotated_frame, f"Actions: {action_text}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update pose history
            prev_poses = current_poses.copy()
            
            # Write frame
            if video_writer:
                video_writer.write(annotated_frame)
        
        # Clean up
        cap.release()
        if video_writer:
            video_writer.release()
        
        print(f"\n✅ Enhanced MMA analysis completed!")
        print(f"📊 Detection Statistics:")
        print(f"   Fighter 1 detections: {detection_stats['fighter_1_detections']}")
        print(f"   Fighter 2 detections: {detection_stats['fighter_2_detections']}")
        print(f"   Referee detections: {detection_stats['referee_detections']}")
        print(f"   Octagon detections: {detection_stats['octagon_detections']}")
        print(f"   Frames with fighters: {detection_stats['total_frames_with_fighters']}")
        print(f"   Strikes detected: {detection_stats['strikes_detected']}")
        print(f"   Takedowns detected: {detection_stats['takedowns_detected']}")
        print(f"   Control time (frames): {detection_stats['control_time']}")
        
        # Save analysis if requested
        if save_analysis:
            self._save_enhanced_analysis(detection_stats, video_path)
        
        return detection_stats
    
    def _draw_pose_keypoints(self, frame, keypoints, fighter_id):
        """Draw pose keypoints on frame"""
        if not keypoints:
            return
        
        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if kp:
                x, y, conf = kp
                color = (0, 255, 0) if fighter_id == "fighter_1" else (255, 0, 0)
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        
        # Draw skeleton connections
        connections = [
            (5, 6),   # shoulders
            (5, 7),   # left shoulder to left elbow
            (7, 9),   # left elbow to left wrist
            (6, 8),   # right shoulder to right elbow
            (8, 10),  # right elbow to right wrist
            (5, 11),  # left shoulder to left hip
            (6, 12),  # right shoulder to right hip
            (11, 12), # hips
            (11, 13), # left hip to left knee
            (13, 15), # left knee to left ankle
            (12, 14), # right hip to right knee
            (14, 16)  # right knee to right ankle
        ]
        
        for connection in connections:
            if (len(keypoints) > max(connection) and 
                keypoints[connection[0]] and keypoints[connection[1]]):
                
                pt1 = keypoints[connection[0]]
                pt2 = keypoints[connection[1]]
                
                color = (0, 255, 0) if fighter_id == "fighter_1" else (255, 0, 0)
                cv2.line(frame, (int(pt1[0]), int(pt1[1])), 
                        (int(pt2[0]), int(pt2[1])), color, 2)
    
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
    
    def _save_enhanced_analysis(self, stats, video_path):
        """Save enhanced analysis to JSON file"""
        analysis = {
            'video_path': video_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': stats,
            'analysis_type': 'enhanced_mma_with_pose'
        }
        
        output_file = f"enhanced_mma_analysis_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"📄 Enhanced analysis saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced MMA Fighter Detection System')
    parser.add_argument('video_path', help='Path to MMA video file')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--model', '-m', help='Path to custom trained MMA model')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, 
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--save-analysis', action='store_true', 
                       help='Save detailed analysis')
    
    args = parser.parse_args()
    
    # Initialize enhanced MMA detector
    detector = EnhancedMMADetector(model_path=args.model)
    
    # Run enhanced detection
    detector.detect_mma_fighters_with_pose(
        video_path=args.video_path,
        output_path=args.output,
        confidence=args.confidence,
        save_analysis=args.save_analysis
    )

if __name__ == "__main__":
    main() 