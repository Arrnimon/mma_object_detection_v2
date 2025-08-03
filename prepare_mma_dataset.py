#!/usr/bin/env python3
"""
MMA Dataset Preparation Script
Helps organize and prepare MMA training data
"""

import os
import cv2
import numpy as np
import json
import shutil
from pathlib import Path
import argparse
from ultralytics import YOLO

class MMADatasetPreparer:
    def __init__(self, dataset_path="mma_dataset"):
        """
        Initialize dataset preparer
        
        Args:
            dataset_path (str): Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.setup_directory_structure()
        
    def setup_directory_structure(self):
        """Create the required directory structure"""
        directories = [
            "train/images", "train/labels",
            "val/images", "val/labels", 
            "test/images", "test/labels"
        ]
        
        for dir_path in directories:
            full_path = self.dataset_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {full_path}")
    
    def extract_frames_from_video(self, video_path, output_dir, frame_interval=30):
        """
        Extract frames from MMA video for annotation
        
        Args:
            video_path (str): Path to MMA video
            output_dir (str): Directory to save frames
            frame_interval (int): Extract every Nth frame
        """
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"🎬 Extracting frames from: {video_path}")
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps}")
        print(f"   Frame interval: {frame_interval}")
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Save frame
                frame_filename = f"frame_{frame_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_count += 1
                
                if saved_count % 10 == 0:
                    print(f"   Saved {saved_count} frames...")
            
            frame_count += 1
        
        cap.release()
        print(f"✅ Extracted {saved_count} frames to {output_dir}")
    
    def create_sample_annotations(self, images_dir, num_samples=10):
        """
        Create sample annotation files for demonstration
        
        Args:
            images_dir (str): Directory containing images
            num_samples (int): Number of sample annotations to create
        """
        images_dir = Path(images_dir)
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        if not image_files:
            print(f"❌ No images found in {images_dir}")
            return
        
        print(f"📝 Creating sample annotations for {min(num_samples, len(image_files))} images...")
        
        for i, image_file in enumerate(image_files[:num_samples]):
            # Read image to get dimensions
            img = cv2.imread(str(image_file))
            if img is None:
                continue
            
            height, width = img.shape[:2]
            
            # Create sample annotation (random bounding boxes for demonstration)
            annotation_lines = []
            
            # Sample fighter 1 (top-left area)
            x1, y1, w1, h1 = 0.2, 0.3, 0.25, 0.4
            annotation_lines.append(f"0 {x1 + w1/2:.6f} {y1 + h1/2:.6f} {w1:.6f} {h1:.6f}")
            
            # Sample fighter 2 (top-right area)
            x2, y2, w2, h2 = 0.6, 0.4, 0.2, 0.35
            annotation_lines.append(f"1 {x2 + w2/2:.6f} {y2 + h2/2:.6f} {w2:.6f} {h2:.6f}")
            
            # Sample referee (bottom area)
            x3, y3, w3, h3 = 0.4, 0.7, 0.15, 0.25
            annotation_lines.append(f"2 {x3 + w3/2:.6f} {y3 + h3/2:.6f} {w3:.6f} {h3:.6f}")
            
            # Sample octagon (full frame)
            annotation_lines.append(f"3 0.5 0.5 0.9 0.9")
            
            # Save annotation file
            annotation_file = image_file.with_suffix('.txt')
            with open(annotation_file, 'w') as f:
                f.write('\n'.join(annotation_lines))
            
            print(f"   Created: {annotation_file.name}")
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        Split dataset into train/val/test sets
        
        Args:
            train_ratio (float): Ratio for training set
            val_ratio (float): Ratio for validation set
            test_ratio (float): Ratio for test set
        """
        if not (abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01):
            print("❌ Ratios must sum to 1.0")
            return
        
        # Get all image files
        all_images = list(self.dataset_path.rglob("*.jpg")) + list(self.dataset_path.rglob("*.png"))
        all_images = [img for img in all_images if "train" not in str(img) and "val" not in str(img) and "test" not in str(img)]
        
        if not all_images:
            print("❌ No images found for splitting")
            return
        
        # Shuffle and split
        np.random.shuffle(all_images)
        
        n_total = len(all_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_images = all_images[:n_train]
        val_images = all_images[n_train:n_train + n_val]
        test_images = all_images[n_train + n_val:]
        
        print(f"📊 Dataset split:")
        print(f"   Total images: {n_total}")
        print(f"   Train: {len(train_images)} ({len(train_images)/n_total:.1%})")
        print(f"   Val: {len(val_images)} ({len(val_images)/n_total:.1%})")
        print(f"   Test: {len(test_images)} ({len(test_images)/n_total:.1%})")
        
        # Move files to appropriate directories
        self._move_files(train_images, "train")
        self._move_files(val_images, "val")
        self._move_files(test_images, "test")
        
        print("✅ Dataset split completed!")
    
    def _move_files(self, image_files, split_name):
        """Move image and label files to split directory"""
        for img_file in image_files:
            # Move image
            dest_img = self.dataset_path / split_name / "images" / img_file.name
            shutil.move(str(img_file), str(dest_img))
            
            # Move corresponding label file
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                dest_label = self.dataset_path / split_name / "labels" / label_file.name
                shutil.move(str(label_file), str(dest_label))
    
    def create_config_file(self):
        """Create YOLO configuration file"""
        config_content = f"""# MMA Dataset Configuration
path: {self.dataset_path.absolute()}  # Dataset root directory
train: train/images  # Train images
val: val/images      # Validation images
test: test/images    # Test images (optional)

# Classes
nc: 7  # Number of classes
names:
  0: fighter_1
  1: fighter_2
  2: referee
  3: octagon
  4: corner
  5: canvas
  6: equipment
"""
        
        config_file = self.dataset_path / "config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Created config file: {config_file}")
    
    def validate_dataset(self):
        """Validate dataset structure and annotations"""
        print("🔍 Validating dataset...")
        
        issues = []
        
        # Check directory structure
        required_dirs = ["train/images", "train/labels", "val/images", "val/labels"]
        for dir_path in required_dirs:
            full_path = self.dataset_path / dir_path
            if not full_path.exists():
                issues.append(f"Missing directory: {dir_path}")
        
        # Check image-label pairs
        for split in ["train", "val"]:
            images_dir = self.dataset_path / split / "images"
            labels_dir = self.dataset_path / split / "labels"
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            label_files = list(labels_dir.glob("*.txt"))
            
            # Check for missing labels
            for img_file in image_files:
                label_file = labels_dir / img_file.with_suffix('.txt').name
                if not label_file.exists():
                    issues.append(f"Missing label for {split}/images/{img_file.name}")
            
            # Check for orphaned labels
            for label_file in label_files:
                img_file = images_dir / label_file.with_suffix('.jpg').name
                if not img_file.exists():
                    img_file = images_dir / label_file.with_suffix('.png').name
                    if not img_file.exists():
                        issues.append(f"Missing image for {split}/labels/{label_file.name}")
        
        if issues:
            print("❌ Dataset validation issues found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Dataset validation passed!")
        
        return len(issues) == 0
    
    def generate_dataset_report(self):
        """Generate a report of dataset statistics"""
        print("📊 Generating dataset report...")
        
        report = {
            "dataset_path": str(self.dataset_path.absolute()),
            "splits": {},
            "total_images": 0,
            "total_annotations": 0
        }
        
        for split in ["train", "val", "test"]:
            images_dir = self.dataset_path / split / "images"
            labels_dir = self.dataset_path / split / "labels"
            
            if not images_dir.exists():
                continue
            
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            label_files = list(labels_dir.glob("*.txt"))
            
            split_stats = {
                "images": len(image_files),
                "labels": len(label_files),
                "class_counts": {i: 0 for i in range(7)}
            }
            
            # Count annotations per class
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                if class_id in split_stats["class_counts"]:
                                    split_stats["class_counts"][class_id] += 1
                except Exception as e:
                    print(f"Warning: Could not read {label_file}: {e}")
            
            report["splits"][split] = split_stats
            report["total_images"] += len(image_files)
            report["total_annotations"] += sum(split_stats["class_counts"].values())
        
        # Save report
        report_file = self.dataset_path / "dataset_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"📈 Dataset Summary:")
        print(f"   Total images: {report['total_images']}")
        print(f"   Total annotations: {report['total_annotations']}")
        
        for split, stats in report["splits"].items():
            print(f"   {split.capitalize()}: {stats['images']} images, {stats['labels']} labels")
        
        print(f"📄 Full report saved to: {report_file}")
        return report

def main():
    parser = argparse.ArgumentParser(description='MMA Dataset Preparation Tool')
    parser.add_argument('--dataset-path', default='mma_dataset', help='Dataset directory path')
    parser.add_argument('--extract-frames', help='Extract frames from video file')
    parser.add_argument('--frame-interval', type=int, default=30, help='Frame extraction interval')
    parser.add_argument('--create-samples', action='store_true', help='Create sample annotations')
    parser.add_argument('--split', action='store_true', help='Split dataset into train/val/test')
    parser.add_argument('--validate', action='store_true', help='Validate dataset')
    parser.add_argument('--report', action='store_true', help='Generate dataset report')
    
    args = parser.parse_args()
    
    # Initialize preparer
    preparer = MMADatasetPreparer(args.dataset_path)
    
    # Extract frames from video
    if args.extract_frames:
        output_dir = preparer.dataset_path / "raw_frames"
        output_dir.mkdir(exist_ok=True)
        preparer.extract_frames_from_video(args.extract_frames, output_dir, args.frame_interval)
        
        # Create sample annotations
        if args.create_samples:
            preparer.create_sample_annotations(output_dir)
    
    # Split dataset
    if args.split:
        preparer.split_dataset()
        preparer.create_config_file()
    
    # Validate dataset
    if args.validate:
        preparer.validate_dataset()
    
    # Generate report
    if args.report:
        preparer.generate_dataset_report()
    
    print("\n🎯 Next steps:")
    print("1. Annotate your images using LabelImg or Roboflow")
    print("2. Run: py prepare_mma_dataset.py --split --validate --report")
    print("3. Start training with: py train_mma_model.py")

if __name__ == "__main__":
    main() 