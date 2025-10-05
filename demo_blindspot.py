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

        # Initialize inference engine with LOWER threshold for better detection
        self.engine = BlindSpotEngine(
            yolo_model_path=model_path,
            midas_model='MiDaS_small',  # Fast for real-time
            conf_threshold=0.15,  # Lower threshold = more detections
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
        last_audio_priority = None  # Track last alert priority

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Process frame
                results = self.engine.process_frame(frame)

                # ADAPTIVE audio feedback - more frequent for critical situations
                spatial_objects = results['spatial_objects']
                current_priority = min([obj.priority for obj in spatial_objects]) if spatial_objects else 4

                # Adaptive frequency based on priority
                if current_priority == 1:  # CRITICAL (< 2.5m) - every 5 frames
                    audio_interval = 5
                elif current_priority == 2:  # WARNING (2.5-4m) - every 10 frames
                    audio_interval = 10
                elif current_priority == 3:  # INFO (4-6m) - every 20 frames
                    audio_interval = 20
                else:  # CLEAR - every 30 frames
                    audio_interval = 30

                # Provide audio feedback
                if audio_enabled and self.audio and frame_count % audio_interval == 0:
                    self._provide_audio_feedback(results)
                    last_audio_priority = current_priority

                # Enhanced visualization with on-screen guidance
                if self.show_visualization:
                    vis_frame = self.engine.visualize_results(results)

                    # Add real-time status overlay
                    self._add_status_overlay(vis_frame, results, current_priority, audio_enabled)

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
                    if self.audio:
                        self.audio.speak(f"Audio {status}")
                elif key == ord('d'):
                    # Describe current scene in detail
                    self._describe_scene(results)
                elif key == ord('h'):
                    # Help/Instructions
                    self._show_help()

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
        """Provide DETAILED audio feedback with enhanced guidance."""
        if not self.audio:
            return

        spatial_objects = results['spatial_objects']
        navigation_hints = results['navigation_hints']

        # Priority 1: CRITICAL alerts (< 2.5m)
        critical = [obj for obj in spatial_objects if obj.priority == 1]
        if critical:
            obj = critical[0]
            # Enhanced critical alert with specific guidance
            self.audio.speak(f"Stop! {obj.class_name} directly ahead at {obj.distance:.1f} meters")
            if len(critical) > 1:
                self.audio.speak(f"Multiple obstacles detected. Use caution.")
            return

        # Priority 2: WARNING alerts (2.5m - 4.0m)
        warnings = [obj for obj in spatial_objects if obj.priority == 2]
        if warnings:
            obj = warnings[0]
            # Detailed warning with position
            position_desc = self._get_position_description(obj.position)
            self.audio.speak(f"Caution. {obj.class_name} {position_desc} at {obj.distance:.1f} meters")

            # Suggest safe direction
            if navigation_hints['safe_direction'] != 'none':
                safe_dir = navigation_hints['safe_direction']
                self.audio.speak(f"Safe path to your {safe_dir}")
            return

        # Priority 3: INFO (4.0m - 6.0m) - describe environment
        info = [obj for obj in spatial_objects if obj.priority == 3]
        if info and len(info) >= 2:
            # Describe multiple objects for better spatial awareness
            obj_descriptions = []
            for obj in info[:3]:  # Top 3 objects
                pos_desc = self._get_position_description(obj.position)
                obj_descriptions.append(f"{obj.class_name} {pos_desc}")

            description = ". ".join(obj_descriptions)
            self.audio.speak(f"Ahead: {description}")
            return

        # No obstacles - encourage forward movement
        if not spatial_objects:
            self.audio.speak("Path clear. You may proceed forward")
        elif navigation_hints['recommendations']:
            recommendation = navigation_hints['recommendations'][0]
            self.audio.speak(recommendation)

    def _get_position_description(self, position):
        """Convert position code to natural language."""
        position_map = {
            'center': 'directly ahead',
            'left': 'on your left',
            'right': 'on your right',
            'far_left': 'far to your left',
            'far_right': 'far to your right'
        }
        return position_map.get(position, position)

    def _add_status_overlay(self, frame, results, priority, audio_enabled):
        """Add real-time status overlay on video frame."""
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        spatial_objects = results['spatial_objects']
        nav_hints = results['navigation_hints']

        # Status panel background (semi-transparent)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Status color based on priority
        color_map = {
            1: (0, 0, 255),    # CRITICAL - Red
            2: (0, 165, 255),  # WARNING - Orange
            3: (0, 255, 255),  # INFO - Yellow
            4: (0, 255, 0)     # CLEAR - Green
        }
        status_color = color_map.get(priority, (255, 255, 255))

        # Status text
        status_text = ["CRITICAL", "WARNING", "INFO", "CLEAR"][priority-1] if priority <= 4 else "UNKNOWN"
        cv2.putText(frame, f"Status: {status_text}", (20, 40),
                    cv2.FONT_HERSHEY_BOLD, 0.8, status_color, 2)

        # Object count
        cv2.putText(frame, f"Objects: {len(spatial_objects)}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Safe direction
        safe_dir = nav_hints['safe_direction'].upper()
        cv2.putText(frame, f"Safe: {safe_dir}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Audio status
        audio_str = "ON" if audio_enabled else "OFF"
        audio_color = (0, 255, 0) if audio_enabled else (128, 128, 128)
        cv2.putText(frame, f"Audio: {audio_str}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 1)

        # FPS
        fps = results['timings'].get('fps', 0)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Help reminder
        cv2.putText(frame, "Press 'h' for help", (w-200, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _describe_scene(self, results):
        """Provide detailed audio description of current scene."""
        if not self.audio:
            return

        spatial_objects = results['spatial_objects']
        nav_hints = results['navigation_hints']

        if not spatial_objects:
            self.audio.speak("No obstacles detected. Path is clear.")
            return

        # Describe all objects by position
        self.audio.speak(f"I detect {len(spatial_objects)} objects in your path.")

        # Group by distance zone
        critical = [o for o in spatial_objects if o.priority == 1]
        warning = [o for o in spatial_objects if o.priority == 2]
        info = [o for o in spatial_objects if o.priority == 3]

        if critical:
            names = ", ".join([o.class_name for o in critical[:3]])
            self.audio.speak(f"Immediate hazard: {names} within 2.5 meters")

        if warning:
            names = ", ".join([o.class_name for o in warning[:3]])
            self.audio.speak(f"Approaching: {names} at 3 to 4 meters")

        if info:
            names = ", ".join([o.class_name for o in info[:3]])
            self.audio.speak(f"Further ahead: {names} at 4 to 6 meters")

        # Safe direction
        if nav_hints['safe_direction'] != 'none':
            self.audio.speak(f"Recommended direction: {nav_hints['safe_direction']}")

    def _show_help(self):
        """Display help information."""
        help_text = """
================================================================================
BlindSpot AI - Enhanced Navigation Assistant
================================================================================

KEYBOARD CONTROLS:
  'q' - Quit application
  's' - Print summary of current scene
  'a' - Toggle audio feedback on/off
  'd' - Describe current scene in detail (audio)
  'h' - Show this help

AUDIO FEEDBACK MODES:
  • CRITICAL (Red): Obstacle < 2.5m - Updates every 5 frames
  • WARNING (Orange): Obstacle 2.5-4.0m - Updates every 10 frames
  • INFO (Yellow): Obstacle 4.0-6.0m - Updates every 20 frames
  • CLEAR (Green): No obstacles - Updates every 30 frames

GUIDANCE FEATURES:
  ✓ Adaptive alert frequency based on danger level
  ✓ Specific object identification (doors, walls, stairs, obstacles)
  ✓ Position awareness (left, right, ahead)
  ✓ Distance estimation in meters
  ✓ Safe path recommendations
  ✓ Multi-object scene descriptions

ON-SCREEN DISPLAY:
  • Color-coded status indicator
  • Object count
  • Safe direction arrow
  • Audio status
  • Real-time FPS

USAGE TIPS:
  1. Use 'd' key frequently for detailed environment scan
  2. Audio adapts automatically - closer objects = more frequent alerts
  3. Follow "Safe path" recommendations when obstacles detected
  4. Press 's' to see text summary of all detected objects

================================================================================
        """
        print(help_text)
        if self.audio:
            self.audio.speak("Help displayed on screen. Check console for controls.")

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
