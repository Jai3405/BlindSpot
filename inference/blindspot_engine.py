"""
BlindSpot Inference Engine
Combines YOLO object detection + MiDaS depth estimation for complete spatial awareness.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import time

from inference.depth_estimator import DepthEstimator
from inference.spatial_analyzer import SpatialAnalyzer, SpatialObject


class BlindSpotEngine:
    """
    Complete BlindSpot AI inference engine.
    Combines object detection, depth estimation, and spatial analysis.
    """

    def __init__(self,
                 yolo_model_path='runs/train/blindspot_optimized/weights/best.pt',
                 midas_model='MiDaS_small',  # Fast model for real-time
                 device=None,
                 conf_threshold=0.25,
                 iou_threshold=0.45):
        """
        Initialize BlindSpot Engine.

        Args:
            yolo_model_path: Path to trained YOLO model
            midas_model: MiDaS model type
            device: 'cuda', 'mps', 'cpu', or None (auto)
            conf_threshold: YOLO confidence threshold
            iou_threshold: YOLO NMS IOU threshold
        """
        print("="*70)
        print("Initializing BlindSpot AI Engine")
        print("="*70)

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device
        print(f"Using device: {self.device}")

        # Load YOLO model
        print(f"\n[1/3] Loading YOLO detector: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        print("✓ YOLO loaded successfully!")

        # Load MiDaS depth estimator
        print(f"\n[2/3] Loading MiDaS depth estimator: {midas_model}")
        self.depth_estimator = DepthEstimator(model_type=midas_model, device=device)
        print("✓ MiDaS loaded successfully!")

        # Initialize spatial analyzer
        print(f"\n[3/3] Initializing spatial analyzer")
        self.spatial_analyzer = SpatialAnalyzer(
            image_width=640,
            image_height=480,
            fov_horizontal=60,
            max_distance=10.0,
            min_distance=0.5
        )
        print("✓ Spatial analyzer ready!")

        print("\n" + "="*70)
        print("BlindSpot AI Engine Ready!")
        print("="*70 + "\n")

    def process_frame(self, frame):
        """
        Process single frame through complete pipeline.

        Args:
            frame: Input image (BGR, numpy array)

        Returns:
            results: Dictionary with all analysis results
        """
        start_time = time.time()

        # Convert BGR to RGB for depth estimation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Step 1: Object Detection
        det_start = time.time()
        yolo_results = self.yolo(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        det_time = time.time() - det_start

        # Extract detections
        detections = []
        if len(yolo_results.boxes) > 0:
            boxes = yolo_results.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                class_name = yolo_results.names[cls]

                detections.append({
                    'bbox': bbox.tolist(),
                    'class': class_name,
                    'conf': conf,
                    'class_id': cls
                })

        # Step 2: Depth Estimation
        depth_start = time.time()
        depth_map = self.depth_estimator.estimate_depth(frame_rgb)
        depth_time = time.time() - depth_start

        # Step 3: Spatial Analysis
        spatial_start = time.time()
        spatial_objects = self.spatial_analyzer.analyze_detections(
            detections, depth_map, depth_method='min'
        )
        navigation_hints = self.spatial_analyzer.get_navigation_hints(spatial_objects)
        spatial_time = time.time() - spatial_start

        total_time = time.time() - start_time

        # Package results
        results = {
            'frame': frame,
            'detections': detections,
            'depth_map': depth_map,
            'spatial_objects': spatial_objects,
            'navigation_hints': navigation_hints,
            'timings': {
                'detection': det_time,
                'depth': depth_time,
                'spatial': spatial_time,
                'total': total_time,
                'fps': 1.0 / total_time if total_time > 0 else 0
            }
        }

        return results

    def visualize_results(self, results, show_depth=True, show_boxes=True):
        """
        Create visualization of results.

        Args:
            results: Results from process_frame()
            show_depth: Show depth map overlay
            show_boxes: Show detection boxes

        Returns:
            vis_frame: Visualized frame
        """
        frame = results['frame'].copy()
        h, w = frame.shape[:2]

        # Draw depth map overlay (semi-transparent)
        if show_depth:
            depth_colored = self.depth_estimator.visualize_depth(
                results['depth_map'],
                colormap=cv2.COLORMAP_INFERNO
            )
            # Resize depth map to match frame
            depth_colored = cv2.resize(depth_colored, (w, h))
            # Blend with original frame
            frame = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)

        # Draw bounding boxes with distance info
        if show_boxes:
            for obj in results['spatial_objects']:
                x1, y1, x2, y2 = [int(v) for v in obj.bbox]

                # Color based on priority
                if obj.priority == 1:
                    color = (0, 0, 255)  # Red - critical
                elif obj.priority == 2:
                    color = (0, 165, 255)  # Orange - warning
                elif obj.priority == 3:
                    color = (0, 255, 255)  # Yellow - info
                else:
                    color = (0, 255, 0)  # Green - low priority

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label with distance
                label = f"{obj.class_name} {obj.distance:.1f}m"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            color, -1)
                cv2.putText(frame, label,
                          (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (255, 255, 255), 1)

        # Draw navigation hints
        hints = results['navigation_hints']

        # Background for text
        cv2.rectangle(frame, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, 120), (255, 255, 255), 2)

        # Draw hints
        y_offset = 30
        cv2.putText(frame, f"Safe Direction: {hints['safe_direction'].upper()}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y_offset += 25
        if hints['critical_alerts']:
            alert_text = "ALERT: " + ", ".join(hints['critical_alerts'][:2])
            cv2.putText(frame, alert_text,
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        y_offset += 20
        if hints['recommendations']:
            rec_text = hints['recommendations'][0]
            cv2.putText(frame, rec_text,
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw FPS
        fps = results['timings']['fps']
        cv2.putText(frame, f"FPS: {fps:.1f}",
                   (w - 120, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def process_image(self, image_path, output_path=None, visualize=True):
        """
        Process single image.

        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            visualize: Create visualization

        Returns:
            results: Processing results
        """
        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Process
        results = self.process_frame(frame)

        # Visualize
        if visualize:
            vis_frame = self.visualize_results(results)

            if output_path:
                cv2.imwrite(str(output_path), vis_frame)
                print(f"Visualization saved to: {output_path}")

            results['visualization'] = vis_frame

        return results

    def process_video(self, input_path, output_path=None, show_live=True):
        """
        Process video file.

        Args:
            input_path: Path to input video
            output_path: Path to save output video (optional)
            show_live: Show live preview

        Returns:
            stats: Processing statistics
        """
        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Setup video writer if output requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        total_time = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                results = self.process_frame(frame)
                vis_frame = self.visualize_results(results)

                # Write to output
                if writer:
                    writer.write(vis_frame)

                # Show live
                if show_live:
                    cv2.imshow('BlindSpot AI', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1
                total_time += results['timings']['total']

                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

        finally:
            cap.release()
            if writer:
                writer.release()
            if show_live:
                cv2.destroyAllWindows()

        stats = {
            'frames_processed': frame_count,
            'total_time': total_time,
            'avg_fps': frame_count / total_time if total_time > 0 else 0
        }

        print(f"\n✓ Video processing complete!")
        print(f"  Frames: {frame_count}")
        print(f"  Average FPS: {stats['avg_fps']:.2f}")

        return stats


def demo_image():
    """Demo with single image."""
    print("BlindSpot AI - Image Demo")
    print("="*70)

    # Initialize engine
    engine = BlindSpotEngine(
        yolo_model_path='runs/train/blindspot_optimized/weights/best.pt',
        midas_model='MiDaS_small'
    )

    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite('test_input.jpg', test_image)

    # Process
    results = engine.process_image('test_input.jpg', 'test_output.jpg')

    print(f"\nDetected {len(results['spatial_objects'])} objects")
    for obj in results['spatial_objects'][:5]:
        print(f"  - {obj.class_name}: {obj.distance:.2f}m ({obj.position})")

    print(f"\nProcessing time: {results['timings']['total']:.3f}s")
    print(f"FPS: {results['timings']['fps']:.1f}")


if __name__ == "__main__":
    demo_image()
