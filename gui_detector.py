import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from enhanced_mma_detector import EnhancedMMADetector
import cv2
from PIL import Image, ImageTk
import numpy as np

class ObjectDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Object Detector")
        self.root.geometry("800x600")
        
        # Initialize detector
        self.detector = None
        self.video_path = None
        self.output_path = None
        self.processing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Video Object Detection", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection
        ttk.Label(main_frame, text="Input Video:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.video_path_var = tk.StringVar()
        video_entry = ttk.Entry(main_frame, textvariable=self.video_path_var, width=50)
        video_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_video).grid(row=1, column=2, pady=5)
        
        # Output file selection
        ttk.Label(main_frame, text="Output Video:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_path_var = tk.StringVar()
        output_entry = ttk.Entry(main_frame, textvariable=self.output_path_var, width=50)
        output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=2, column=2, pady=5)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Detection Settings", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        settings_frame.columnconfigure(1, weight=1)
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        self.confidence_label = ttk.Label(settings_frame, text="0.5")
        self.confidence_label.grid(row=0, column=2, pady=5)
        confidence_scale.configure(command=self.update_confidence_label)
        
        # MMA-specific options
        self.track_fighters_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Track individual fighters", 
                       variable=self.track_fighters_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Options
        self.save_frames_var = tk.BooleanVar()
        ttk.Checkbutton(settings_frame, text="Save individual frames", 
                       variable=self.save_frames_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.progress_var).grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Log text
        self.log_text = tk.Text(progress_frame, height=8, width=70)
        self.log_text.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Scrollbar for log
        scrollbar = ttk.Scrollbar(progress_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=2, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=5, column=0, columnspan=3, pady=20)
        
        self.detect_button = ttk.Button(buttons_frame, text="Detect Objects", 
                                       command=self.start_detection)
        self.detect_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.analyze_button = ttk.Button(buttons_frame, text="Analyze Video", 
                                        command=self.start_analysis)
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(buttons_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT)
        
    def update_confidence_label(self, value):
        self.confidence_label.config(text=f"{float(value):.1f}")
        
    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.video_path_var.set(filename)
            self.video_path = filename
            
            # Auto-generate output path
            base_name = os.path.splitext(filename)[0]
            self.output_path_var.set(f"{base_name}_detected.mp4")
            self.output_path = f"{base_name}_detected.mp4"
            
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save Output Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if filename:
            self.output_path_var.set(filename)
            self.output_path = filename
            
    def log_message(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        
    def start_detection(self):
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showerror("Error", "Please select a valid video file")
            return
            
        if self.processing:
            messagebox.showwarning("Warning", "Detection already in progress")
            return
            
        # Get settings
        confidence = self.confidence_var.get()
        target_classes = None  # Use all available classes
        save_frames = self.save_frames_var.get()
        track_fighters = self.track_fighters_var.get()
        
        # Start detection in separate thread
        self.processing = True
        self.progress_bar.start()
        self.detect_button.config(state='disabled')
        self.analyze_button.config(state='disabled')
        
        thread = threading.Thread(target=self.run_detection, 
                                args=(confidence, target_classes, save_frames, track_fighters))
        thread.daemon = True
        thread.start()
        
    def run_detection(self, confidence, target_classes, save_frames):
        try:
            self.log_message("Initializing enhanced MMA detector...")
            self.detector = EnhancedMMADetector()
            
            self.log_message(f"Starting enhanced detection with confidence threshold: {confidence}")
            if target_classes:
                self.log_message(f"Target classes: {', '.join(target_classes)}")
            
            # Create output directory for frames if needed
            output_dir = None
            if save_frames:
                output_dir = os.path.splitext(self.output_path)[0] + "_frames"
                os.makedirs(output_dir, exist_ok=True)
                self.log_message(f"Saving frames to: {output_dir}")
            
            # Run enhanced detection
            stats = self.detector.detect_mma_fighters_with_pose(
                video_path=self.video_path,
                output_path=self.output_path,
                confidence=confidence,
                target_classes=target_classes,
                save_frames=save_frames,
                output_dir=output_dir
            )
            
            self.log_message("Detection completed successfully!")
            self.log_message(f"Output saved to: {self.output_path}")
            
            if stats:
                self.log_message("Detection statistics:")
                for class_name, count in stats.items():
                    self.log_message(f"  {class_name}: {count} detections")
                    
        except Exception as e:
            self.log_message(f"Error during detection: {str(e)}")
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            
        finally:
            self.processing = False
            self.progress_bar.stop()
            self.detect_button.config(state='normal')
            self.analyze_button.config(state='normal')
            
    def start_analysis(self):
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showerror("Error", "Please select a valid video file")
            return
            
        if self.processing:
            messagebox.showwarning("Warning", "Analysis already in progress")
            return
            
        # Get settings
        confidence = self.confidence_var.get()
        classes_str = self.classes_var.get().strip()
        target_classes = classes_str.split() if classes_str else None
        
        # Start analysis in separate thread
        self.processing = True
        self.progress_bar.start()
        self.detect_button.config(state='disabled')
        self.analyze_button.config(state='disabled')
        
        thread = threading.Thread(target=self.run_analysis, 
                                args=(confidence, target_classes))
        thread.daemon = True
        thread.start()
        
    def run_analysis(self, confidence, target_classes):
        try:
            self.log_message("Initializing enhanced MMA detector...")
            self.detector = EnhancedMMADetector()
            
            self.log_message(f"Starting enhanced analysis with confidence threshold: {confidence}")
            if target_classes:
                self.log_message(f"Target classes: {', '.join(target_classes)}")
            
            # Run enhanced analysis
            analysis = self.detector.detect_mma_fighters_with_pose(
                video_path=self.video_path,
                confidence=confidence,
                target_classes=target_classes
            )
            
            self.log_message("Analysis completed!")
            self.log_message(f"Video duration: {analysis['duration']:.2f} seconds")
            self.log_message(f"Total detections: {len(analysis['detections'])}")
            
            # Group detections by class
            class_counts = {}
            for detection in analysis['detections']:
                class_name = detection['class']
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
            
            if class_counts:
                self.log_message("Detections by class:")
                for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    self.log_message(f"  {class_name}: {count}")
                    
        except Exception as e:
            self.log_message(f"Error during analysis: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            
        finally:
            self.processing = False
            self.progress_bar.stop()
            self.detect_button.config(state='normal')
            self.analyze_button.config(state='normal')

def main():
    root = tk.Tk()
    app = ObjectDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 