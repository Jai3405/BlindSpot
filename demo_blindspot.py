#!/usr/bin/env python3
"""
BlindSpot AI - Complete Demo
Demonstrates the full BlindSpot AI system with webcam/video support.

Usage:
    python demo_blindspot.py --mode webcam          # Live webcam
    python demo_blindspot.py --mode video --input path/to/video.mp4
    python demo_blindspot.py --mode image --input path/to/image.jpg
"""

import argparse
import cv2
import sys
from pathlib import Path

from inference.blindspot_engine import BlindSpotEngine
from inference.audio_feedback import AudioFeedback


class BlindSpotDemo:
    """Complete BlindSpot AI demonstration."""

    def __init__(self,
                 model_path='runs/train/blindspot_optimized/weights/best.pt',
                 enable_audio=True,
                 show_visualization=True):
        """
        Initialize BlindSpot demo.

        Args:
            model_path: Path to trained YOLO model
            enable_audio: Enable audio feedback
            show_visualization: Show visual output
        """
        print("\n" + "="*70)
        print("BlindSpot AI - Assistive Navigation System")
        print("="*70 + "\n")

        # Initialize inference engine
        self.engine = BlindSpotEngine(
            yolo_model_path=model_path,
            midas_model='MiDaS_small',  # Fast for real-time
            conf_threshold=0.35,
            iou_threshold=0.45
        )

        # Initialize audio feedback
        self.audio = None
        if enable_audio:
            try:
                self.audio = AudioFeedback(rate=180, volume=0.9)
                print("\n✓ Audio feedback enabled")
            except Exception as e:
                print(f"\n⚠️  Audio feedback disabled: {e}")
                self.audio = None

        self.show_visualization = show_visualization

        print("\n" + "="*70)
        print("System Ready!")
        print("="*70 + "\n")

    def run_webcam(self, camera_id=0):
        """
        Run BlindSpot on live webcam feed.

        Args:
            camera_id: Camera device ID (usually 0)
        """
        print(f"Starting webcam (camera {camera_id})...")
        print("Press 'q' to quit, 's' for summary, 'a' to toggle audio\n")

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        frame_count = 0
        audio_enabled = self.audio is not None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Process frame
                results = self.engine.process_frame(frame)

                # Audio feedback (periodic, not every frame)
                if audio_enabled and self.audio and frame_count % 30 == 0:
                    self._provide_audio_feedback(results)

                # Visualization
                if self.show_visualization:
                    vis_frame = self.engine.visualize_results(results)
                    cv2.imshow('BlindSpot AI', vis_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    self._print_summary(results)
                elif key == ord('a'):
                    audio_enabled = not audio_enabled
                    status = "enabled" if audio_enabled else "disabled"
                    print(f"\nAudio feedback {status}")

                frame_count += 1

        finally:
            cap.release()
            if self.show_visualization:
                cv2.destroyAllWindows()

    def run_video(self, video_path, output_path=None):
        """
        Run BlindSpot on video file.

        Args:
            video_path: Path to input video
            output_path: Path to save output (optional)
        """
        print(f"Processing video: {video_path}")

        if output_path:
            print(f"Output will be saved to: {output_path}")

        stats = self.engine.process_video(
            video_path,
            output_path=output_path,
            show_live=self.show_visualization
        )

        print(f"\n✓ Video processing complete!")
        print(f"Average FPS: {stats['avg_fps']:.2f}")

    def run_image(self, image_path, output_path=None):
        """
        Run BlindSpot on single image.

        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
        """
        print(f"Processing image: {image_path}")

        results = self.engine.process_image(
            image_path,
            output_path=output_path,
            visualize=True
        )

        # Print results
        self._print_summary(results)

        # Show visualization
        if self.show_visualization and 'visualization' in results:
            cv2.imshow('BlindSpot AI - Press any key to close', results['visualization'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Audio feedback
        if self.audio:
            self._provide_audio_feedback(results)

    def _provide_audio_feedback(self, results):
        """Provide audio feedback based on results."""
        if not self.audio:
            return

        spatial_objects = results['spatial_objects']
        navigation_hints = results['navigation_hints']

        # Critical alerts (priority 1)
        critical = [obj for obj in spatial_objects if obj.priority == 1]
        if critical:
            self.audio.announce_critical_alert(critical[0])
            return

        # Navigation hints
        if navigation_hints['recommendations']:
            recommendation = navigation_hints['recommendations'][0]
            if "clear" not in recommendation.lower():
                self.audio.speak(recommendation)

    def _print_summary(self, results):
        """Print text summary of results."""
        spatial_objects = results['spatial_objects']
        navigation_hints = results['navigation_hints']

        print("\n" + "="*70)
        print(f"Detected {len(spatial_objects)} objects:")

        for i, obj in enumerate(spatial_objects[:5]):  # Top 5
            priority_label = ["CRITICAL", "WARNING", "INFO", "LOW"][obj.priority - 1]
            print(f"  [{priority_label}] {obj.class_name} - {obj.position.upper()} - {obj.distance:.2f}m")

        print(f"\nSafe Direction: {navigation_hints['safe_direction'].upper()}")

        if navigation_hints['recommendations']:
            print(f"Recommendation: {navigation_hints['recommendations'][0]}")

        print(f"\nFPS: {results['timings']['fps']:.1f}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='BlindSpot AI Demo')

    parser.add_argument('--mode', type=str, required=True,
                        choices=['webcam', 'video', 'image'],
                        help='Operating mode')

    parser.add_argument('--input', type=str, default=None,
                        help='Input file path (for video/image mode)')

    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (optional)')

    parser.add_argument('--model', type=str,
                        default='runs/train/blindspot_optimized/weights/best.pt',
                        help='Path to YOLO model')

    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID for webcam mode')

    parser.add_argument('--no-audio', action='store_true',
                        help='Disable audio feedback')

    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization')

    args = parser.parse_args()

    # Validate inputs
    if args.mode in ['video', 'image'] and args.input is None:
        print(f"Error: --input required for {args.mode} mode")
        sys.exit(1)

    if args.input and not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print("Please train the model first using model_training/train_yolov8.py")
        sys.exit(1)

    # Initialize demo
    demo = BlindSpotDemo(
        model_path=args.model,
        enable_audio=not args.no_audio,
        show_visualization=not args.no_viz
    )

    # Run appropriate mode
    try:
        if args.mode == 'webcam':
            demo.run_webcam(camera_id=args.camera)

        elif args.mode == 'video':
            demo.run_video(args.input, output_path=args.output)

        elif args.mode == 'image':
            demo.run_image(args.input, output_path=args.output)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nBlindSpot AI shut down cleanly.")


if __name__ == "__main__":
    main()
