#!/usr/bin/env python3
"""
Video to Frames Extractor for BlindSpot
Extracts high-quality frames from videos for custom dataset collection
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """Extract frames from videos with quality filtering"""

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        fps: float = 2.0,
        blur_threshold: float = 100.0,
        min_brightness: int = 20,
        max_brightness: int = 235
    ):
        """
        Initialize frame extractor

        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            fps: Frames per second to extract (default 2.0)
            blur_threshold: Laplacian variance threshold for blur detection
            min_brightness: Minimum average brightness
            max_brightness: Maximum average brightness
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'video_path': str(video_path),
            'total_frames': 0,
            'extracted_frames': 0,
            'blurry_frames': 0,
            'too_dark_frames': 0,
            'too_bright_frames': 0,
            'saved_frames': 0
        }

    def calculate_blur(self, image: np.ndarray) -> float:
        """
        Calculate blur score using Laplacian variance

        Args:
            image: Input image (BGR)

        Returns:
            Blur score (higher = sharper)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def calculate_brightness(self, image: np.ndarray) -> float:
        """
        Calculate average brightness

        Args:
            image: Input image (BGR)

        Returns:
            Average brightness (0-255)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def is_frame_quality_good(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Check if frame meets quality criteria

        Args:
            frame: Input frame

        Returns:
            (is_good, reason)
        """
        # Check blur
        blur_score = self.calculate_blur(frame)
        if blur_score < self.blur_threshold:
            return False, f'blurry (score: {blur_score:.1f})'

        # Check brightness
        brightness = self.calculate_brightness(frame)
        if brightness < self.min_brightness:
            return False, f'too dark (brightness: {brightness:.1f})'
        if brightness > self.max_brightness:
            return False, f'too bright (brightness: {brightness:.1f})'

        return True, 'good'

    def extract_frames(self):
        """Extract frames from video"""
        if not self.video_path.exists():
            logger.error(f"Video file not found: {self.video_path}")
            return

        logger.info(f"Extracting frames from: {self.video_path.name}")

        # Open video
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            return

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        logger.info(f"Video properties:")
        logger.info(f"  FPS: {video_fps:.2f}")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.info(f"  Extraction rate: {self.fps} FPS")
        logger.info(f"  Expected frames: {int(duration * self.fps)}")

        self.stats['total_frames'] = total_frames

        # Calculate frame skip
        frame_skip = int(video_fps / self.fps)
        if frame_skip < 1:
            frame_skip = 1

        logger.info(f"  Frame skip: {frame_skip} (extracting every {frame_skip}th frame)\n")

        # Extract frames
        frame_count = 0
        saved_count = 0

        # Create video-specific subdirectory
        video_name = self.video_path.stem
        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(exist_ok=True)

        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Only process frames at desired FPS
                if frame_count % frame_skip == 0:
                    self.stats['extracted_frames'] += 1

                    # Check frame quality
                    is_good, reason = self.is_frame_quality_good(frame)

                    if is_good:
                        # Save frame
                        frame_filename = f"{video_name}_frame_{saved_count:06d}.jpg"
                        frame_path = video_output_dir / frame_filename

                        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        saved_count += 1
                        self.stats['saved_frames'] += 1

                    else:
                        # Track rejection reason
                        if 'blurry' in reason:
                            self.stats['blurry_frames'] += 1
                        elif 'dark' in reason:
                            self.stats['too_dark_frames'] += 1
                        elif 'bright' in reason:
                            self.stats['too_bright_frames'] += 1

                frame_count += 1
                pbar.update(1)

        cap.release()

        logger.info(f"\n✓ Extraction complete!")
        logger.info(f"  Frames saved to: {video_output_dir}")
        logger.info(f"  Total saved: {saved_count}")

    def generate_statistics(self):
        """Generate extraction statistics"""
        stats_file = self.output_dir / f'{self.video_path.stem}_stats.json'

        self.stats['extraction_date'] = datetime.now().isoformat()

        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

        logger.info("\n" + "="*60)
        logger.info("Extraction Statistics")
        logger.info("="*60)
        logger.info(f"Total frames in video: {self.stats['total_frames']}")
        logger.info(f"Extracted frames (at {self.fps} FPS): {self.stats['extracted_frames']}")
        logger.info(f"Saved frames: {self.stats['saved_frames']}")
        logger.info(f"\nRejected frames:")
        logger.info(f"  Blurry: {self.stats['blurry_frames']}")
        logger.info(f"  Too dark: {self.stats['too_dark_frames']}")
        logger.info(f"  Too bright: {self.stats['too_bright_frames']}")
        logger.info(f"\nAcceptance rate: {self.stats['saved_frames']/self.stats['extracted_frames']*100:.1f}%")
        logger.info(f"\n✓ Statistics saved to {stats_file}")

    def run(self):
        """Run extraction pipeline"""
        self.extract_frames()
        self.generate_statistics()


class BatchVideoExtractor:
    """Extract frames from multiple videos"""

    def __init__(
        self,
        video_dir: str,
        output_dir: str,
        fps: float = 2.0,
        blur_threshold: float = 100.0
    ):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.blur_threshold = blur_threshold

    def find_videos(self) -> List[Path]:
        """Find all video files in directory"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        videos = []

        for ext in video_extensions:
            videos.extend(self.video_dir.glob(f'*{ext}'))
            videos.extend(self.video_dir.glob(f'*{ext.upper()}'))

        return sorted(videos)

    def run(self):
        """Process all videos"""
        videos = self.find_videos()

        if not videos:
            logger.warning(f"No videos found in {self.video_dir}")
            return

        logger.info(f"Found {len(videos)} videos to process")

        for i, video_path in enumerate(videos, 1):
            logger.info("\n" + "="*60)
            logger.info(f"Processing video {i}/{len(videos)}")
            logger.info("="*60 + "\n")

            extractor = VideoFrameExtractor(
                video_path=video_path,
                output_dir=self.output_dir,
                fps=self.fps,
                blur_threshold=self.blur_threshold
            )
            extractor.run()

        logger.info("\n" + "="*60)
        logger.info("✓ Batch Processing Complete!")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from videos for BlindSpot dataset'
    )
    parser.add_argument(
        '--video',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        help='Directory containing multiple videos (batch mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/custom',
        help='Output directory for frames'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=2.0,
        help='Frames per second to extract (default: 2.0)'
    )
    parser.add_argument(
        '--blur-threshold',
        type=float,
        default=100.0,
        help='Blur detection threshold (higher = stricter, default: 100)'
    )
    parser.add_argument(
        '--min-brightness',
        type=int,
        default=20,
        help='Minimum average brightness (0-255, default: 20)'
    )
    parser.add_argument(
        '--max-brightness',
        type=int,
        default=235,
        help='Maximum average brightness (0-255, default: 235)'
    )

    args = parser.parse_args()

    if not args.video and not args.video_dir:
        parser.error("Either --video or --video-dir must be specified")

    # Batch mode
    if args.video_dir:
        extractor = BatchVideoExtractor(
            video_dir=args.video_dir,
            output_dir=args.output,
            fps=args.fps,
            blur_threshold=args.blur_threshold
        )
        extractor.run()

    # Single video mode
    elif args.video:
        extractor = VideoFrameExtractor(
            video_path=args.video,
            output_dir=args.output,
            fps=args.fps,
            blur_threshold=args.blur_threshold,
            min_brightness=args.min_brightness,
            max_brightness=args.max_brightness
        )
        extractor.run()


if __name__ == '__main__':
    main()
