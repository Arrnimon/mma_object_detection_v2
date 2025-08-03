#!/usr/bin/env python3
"""
Batch processor for multiple video files
"""

import os
import glob
from pathlib import Path
from object_detector import VideoObjectDetector
import argparse
import time

class BatchVideoProcessor:
    def __init__(self, input_dir, output_dir, confidence=0.5, target_classes=None):
        """
        Initialize batch processor
        
        Args:
            input_dir (str): Directory containing input videos
            output_dir (str): Directory to save output videos
            confidence (float): Confidence threshold
            target_classes (list): Target classes to detect
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.confidence = confidence
        self.target_classes = target_classes
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detector
        self.detector = VideoObjectDetector()
        
        # Supported video extensions
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        
    def get_video_files(self):
        """Get all video files from input directory"""
        video_files = []
        
        for ext in self.video_extensions:
            pattern = str(self.input_dir / f"*{ext}")
            video_files.extend(glob.glob(pattern, recursive=True))
            pattern = str(self.input_dir / f"*{ext.upper()}")
            video_files.extend(glob.glob(pattern, recursive=True))
        
        return sorted(video_files)
    
    def process_video(self, video_path):
        """Process a single video file"""
        video_path = Path(video_path)
        
        # Generate output path
        output_name = f"{video_path.stem}_detected{video_path.suffix}"
        output_path = self.output_dir / output_name
        
        print(f"Processing: {video_path.name}")
        print(f"Output: {output_path.name}")
        
        try:
            # Process video
            stats = self.detector.detect_objects_in_video(
                video_path=str(video_path),
                output_path=str(output_path),
                confidence=self.confidence,
                target_classes=self.target_classes
            )
            
            print(f"✓ Completed: {video_path.name}")
            if stats:
                print(f"  Objects detected: {sum(stats.values())}")
                for class_name, count in stats.items():
                    print(f"    {class_name}: {count}")
            
            return True, stats
            
        except Exception as e:
            print(f"✗ Error processing {video_path.name}: {str(e)}")
            return False, None
    
    def process_all(self, save_frames=False, frames_dir=None):
        """Process all video files in the input directory"""
        video_files = self.get_video_files()
        
        if not video_files:
            print(f"No video files found in {self.input_dir}")
            return
        
        print(f"Found {len(video_files)} video files to process")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Confidence threshold: {self.confidence}")
        if self.target_classes:
            print(f"Target classes: {', '.join(self.target_classes)}")
        print("-" * 50)
        
        # Process each video
        successful = 0
        failed = 0
        total_stats = {}
        
        start_time = time.time()
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] Processing: {Path(video_path).name}")
            
            success, stats = self.process_video(video_path)
            
            if success:
                successful += 1
                # Aggregate statistics
                if stats:
                    for class_name, count in stats.items():
                        if class_name not in total_stats:
                            total_stats[class_name] = 0
                        total_stats[class_name] += count
            else:
                failed += 1
        
        # Print summary
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n" + "=" * 50)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total videos: {len(video_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Average time per video: {processing_time/len(video_files):.2f} seconds")
        
        if total_stats:
            print(f"\nTotal objects detected: {sum(total_stats.values())}")
            print("Objects by class:")
            for class_name, count in sorted(total_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {count}")
        
        print(f"\nOutput saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Batch process multiple video files')
    parser.add_argument('input_dir', help='Directory containing input videos')
    parser.add_argument('output_dir', help='Directory to save output videos')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--classes', nargs='+', help='Target classes to detect')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save individual frames with detections')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Create processor
    processor = BatchVideoProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        confidence=args.confidence,
        target_classes=args.classes
    )
    
    # Process all videos
    processor.process_all(save_frames=args.save_frames)

if __name__ == "__main__":
    main() 